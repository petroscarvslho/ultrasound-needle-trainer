#!/usr/bin/env python3
"""
NEEDLE PILOT - Script de Inferencia
Detecta a ponta da agulha em imagens de ultrassom usando o modelo VASST treinado.

Uso:
    python inference.py --image path/to/image.png
    python inference.py --folder path/to/images/
    python inference.py --video path/to/video.mp4
    python inference.py --camera 0  # Webcam
"""

import argparse
import sys
import time
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import json

import numpy as np
import torch
import torch.nn as nn
import cv2

# Importar modelo do train_vasst
from train_vasst import VASSTPyTorch

# Constantes
MODEL_PATH = Path(__file__).parent / "models" / "vasst_needle.pt"
INPUT_SIZE = (256, 256)
CONFIDENCE_THRESHOLD = 0.5


class NeedleDetector:
    """Detector de agulhas em imagens de ultrassom"""

    def __init__(self, model_path: Path = MODEL_PATH, device: str = None):
        """
        Inicializa o detector

        Args:
            model_path: Caminho para o modelo .pt
            device: Device ('cuda', 'mps', 'cpu' ou None para auto)
        """
        self.model_path = Path(model_path)
        self.input_size = INPUT_SIZE

        # Configurar device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        # Carregar modelo
        self._load_model()

    def _load_model(self):
        """Carrega o modelo treinado"""
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Modelo nao encontrado: {self.model_path}\n"
                "Execute train_vasst.py primeiro para treinar o modelo."
            )

        self.model = VASSTPyTorch(input_shape=self.input_size)
        checkpoint = torch.load(self.model_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model = self.model.to(self.device)
        self.model.eval()

        # Metadata do modelo
        self.model_epoch = checkpoint.get('epoch', 'N/A')
        self.model_val_loss = checkpoint.get('val_loss', 'N/A')

        print(f"Modelo carregado: {self.model_path}")
        print(f"  Device: {self.device}")
        print(f"  Epoca: {self.model_epoch}")
        print(f"  Val Loss: {self.model_val_loss:.6f}" if isinstance(self.model_val_loss, float) else f"  Val Loss: {self.model_val_loss}")

    def preprocess(self, image: np.ndarray) -> Tuple[torch.Tensor, Tuple[int, int]]:
        """
        Pre-processa imagem para inferencia

        Args:
            image: Imagem BGR ou grayscale (numpy array)

        Returns:
            tensor: Tensor pronto para o modelo
            original_size: Tamanho original (height, width)
        """
        original_size = image.shape[:2]

        # Converter para grayscale se necessario
        if len(image.shape) == 3 and image.shape[2] == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image

        # Redimensionar
        resized = cv2.resize(gray, self.input_size, interpolation=cv2.INTER_AREA)

        # Normalizar e converter para tensor
        tensor = torch.FloatTensor(resized).unsqueeze(0).unsqueeze(0) / 255.0
        tensor = tensor.to(self.device)

        return tensor, original_size

    def predict(self, image: np.ndarray) -> Dict:
        """
        Detecta a ponta da agulha na imagem

        Args:
            image: Imagem BGR ou grayscale

        Returns:
            dict com:
                - tip_x, tip_y: Coordenadas da ponta (pixels originais)
                - tip_norm: Coordenadas normalizadas [0,1]
                - confidence: Confianca estimada
                - inference_time_ms: Tempo de inferencia
        """
        start_time = time.time()

        # Preprocessar
        tensor, original_size = self.preprocess(image)

        # Inferencia
        with torch.no_grad():
            output = self.model(tensor)

        # Pos-processar
        pred = output.cpu().numpy()[0]
        tip_y_norm, tip_x_norm = pred

        # Clamp para [0, 1]
        tip_y_norm = np.clip(tip_y_norm, 0, 1)
        tip_x_norm = np.clip(tip_x_norm, 0, 1)

        # Converter para coordenadas originais
        tip_y = int(tip_y_norm * original_size[0])
        tip_x = int(tip_x_norm * original_size[1])

        inference_time = (time.time() - start_time) * 1000

        return {
            'tip_x': tip_x,
            'tip_y': tip_y,
            'tip_norm': [float(tip_y_norm), float(tip_x_norm)],
            'original_size': original_size,
            'inference_time_ms': inference_time
        }

    def draw_prediction(
        self,
        image: np.ndarray,
        prediction: Dict,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
    ) -> np.ndarray:
        """
        Desenha a predicao na imagem

        Args:
            image: Imagem original
            prediction: Resultado de predict()
            color: Cor BGR do marcador
            thickness: Espessura das linhas

        Returns:
            Imagem com anotacoes
        """
        output = image.copy()
        if len(output.shape) == 2:
            output = cv2.cvtColor(output, cv2.COLOR_GRAY2BGR)

        tip_x = prediction['tip_x']
        tip_y = prediction['tip_y']

        # Desenhar crosshair
        size = 15
        cv2.line(output, (tip_x - size, tip_y), (tip_x + size, tip_y), color, thickness)
        cv2.line(output, (tip_x, tip_y - size), (tip_x, tip_y + size), color, thickness)

        # Circulo
        cv2.circle(output, (tip_x, tip_y), size, color, thickness)

        # Texto com coordenadas
        text = f"Tip: ({tip_x}, {tip_y})"
        cv2.putText(output, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Tempo de inferencia
        time_text = f"{prediction['inference_time_ms']:.1f}ms"
        cv2.putText(output, time_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        return output


def process_image(detector: NeedleDetector, image_path: str, output_path: str = None, show: bool = True):
    """Processa uma unica imagem"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Erro ao ler imagem: {image_path}")
        return None

    prediction = detector.predict(image)
    print(f"\nResultado para {image_path}:")
    print(f"  Ponta da agulha: ({prediction['tip_x']}, {prediction['tip_y']})")
    print(f"  Tempo: {prediction['inference_time_ms']:.1f}ms")

    annotated = detector.draw_prediction(image, prediction)

    if output_path:
        cv2.imwrite(output_path, annotated)
        print(f"  Salvo em: {output_path}")

    if show:
        cv2.imshow("Needle Detection", annotated)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    return prediction


def process_folder(detector: NeedleDetector, folder_path: str, output_folder: str = None):
    """Processa todas as imagens de uma pasta"""
    folder = Path(folder_path)
    extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}

    images = [f for f in folder.iterdir() if f.suffix.lower() in extensions]

    if not images:
        print(f"Nenhuma imagem encontrada em {folder}")
        return []

    if output_folder:
        out_path = Path(output_folder)
        out_path.mkdir(parents=True, exist_ok=True)
    else:
        out_path = None

    results = []
    total_time = 0

    print(f"\nProcessando {len(images)} imagens...")

    for img_path in images:
        image = cv2.imread(str(img_path))
        if image is None:
            continue

        prediction = detector.predict(image)
        prediction['file'] = img_path.name
        results.append(prediction)
        total_time += prediction['inference_time_ms']

        if out_path:
            annotated = detector.draw_prediction(image, prediction)
            cv2.imwrite(str(out_path / img_path.name), annotated)

    avg_time = total_time / len(results) if results else 0
    print(f"\nProcessadas: {len(results)} imagens")
    print(f"Tempo medio: {avg_time:.1f}ms")
    print(f"FPS estimado: {1000/avg_time:.1f}" if avg_time > 0 else "")

    # Salvar resultados em JSON
    if out_path:
        json_path = out_path / "results.json"
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Resultados salvos em: {json_path}")

    return results


def process_video(detector: NeedleDetector, video_path: str, output_path: str = None):
    """Processa um video"""
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print(f"Erro ao abrir video: {video_path}")
        return

    # Propriedades do video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"\nVideo: {video_path}")
    print(f"  Resolucao: {width}x{height}")
    print(f"  FPS: {fps}")
    print(f"  Frames: {total_frames}")

    # Writer para output
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_count = 0
    total_time = 0

    print("\nPressione 'q' para sair...")

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prediction = detector.predict(frame)
        total_time += prediction['inference_time_ms']
        frame_count += 1

        annotated = detector.draw_prediction(frame, prediction)

        # Adicionar progresso
        progress = f"Frame {frame_count}/{total_frames}"
        cv2.putText(annotated, progress, (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        if writer:
            writer.write(annotated)

        cv2.imshow("Needle Detection - Video", annotated)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
        print(f"\nVideo salvo em: {output_path}")
    cv2.destroyAllWindows()

    avg_time = total_time / frame_count if frame_count > 0 else 0
    print(f"\nProcessados: {frame_count} frames")
    print(f"Tempo medio: {avg_time:.1f}ms/frame")
    print(f"FPS de processamento: {1000/avg_time:.1f}" if avg_time > 0 else "")


def process_camera(detector: NeedleDetector, camera_id: int = 0):
    """Processa stream de camera em tempo real"""
    cap = cv2.VideoCapture(camera_id)

    if not cap.isOpened():
        print(f"Erro ao abrir camera {camera_id}")
        return

    print(f"\nCamera {camera_id} iniciada")
    print("Pressione 'q' para sair, 's' para salvar frame...")

    frame_count = 0
    fps_counter = 0
    fps_time = time.time()
    display_fps = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        prediction = detector.predict(frame)
        annotated = detector.draw_prediction(frame, prediction, color=(0, 255, 0))

        # Calcular FPS
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            display_fps = fps_counter
            fps_counter = 0
            fps_time = time.time()

        # Mostrar FPS
        cv2.putText(annotated, f"FPS: {display_fps}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        cv2.imshow("Needle Detection - Camera", annotated)

        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            save_path = f"capture_{frame_count}.png"
            cv2.imwrite(save_path, annotated)
            print(f"Frame salvo: {save_path}")
            frame_count += 1

    cap.release()
    cv2.destroyAllWindows()


def benchmark(detector: NeedleDetector, n_iterations: int = 100):
    """Executa benchmark de performance"""
    print(f"\nExecutando benchmark ({n_iterations} iteracoes)...")

    # Criar imagem sintetica
    test_image = np.random.randint(0, 255, (256, 256), dtype=np.uint8)

    # Warmup
    for _ in range(10):
        detector.predict(test_image)

    # Benchmark
    times = []
    for _ in range(n_iterations):
        start = time.time()
        detector.predict(test_image)
        times.append((time.time() - start) * 1000)

    times = np.array(times)
    print(f"\nResultados do Benchmark:")
    print(f"  Media: {np.mean(times):.2f}ms")
    print(f"  Std: {np.std(times):.2f}ms")
    print(f"  Min: {np.min(times):.2f}ms")
    print(f"  Max: {np.max(times):.2f}ms")
    print(f"  P50: {np.percentile(times, 50):.2f}ms")
    print(f"  P95: {np.percentile(times, 95):.2f}ms")
    print(f"  P99: {np.percentile(times, 99):.2f}ms")
    print(f"  FPS teorico: {1000/np.mean(times):.1f}")


def main():
    parser = argparse.ArgumentParser(
        description="NEEDLE PILOT - Deteccao de Agulhas em Ultrassom",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemplos:
  python inference.py --image ultrasound.png
  python inference.py --folder ./images --output ./results
  python inference.py --video procedure.mp4 --output output.mp4
  python inference.py --camera 0
  python inference.py --benchmark
        """
    )

    parser.add_argument('--image', '-i', type=str, help='Caminho para imagem')
    parser.add_argument('--folder', '-f', type=str, help='Pasta com imagens')
    parser.add_argument('--video', '-v', type=str, help='Caminho para video')
    parser.add_argument('--camera', '-c', type=int, help='ID da camera (0, 1, ...)')
    parser.add_argument('--output', '-o', type=str, help='Caminho de saida')
    parser.add_argument('--model', '-m', type=str, default=str(MODEL_PATH), help='Caminho do modelo')
    parser.add_argument('--device', '-d', type=str, choices=['cuda', 'mps', 'cpu'], help='Device')
    parser.add_argument('--benchmark', '-b', action='store_true', help='Executar benchmark')
    parser.add_argument('--no-display', action='store_true', help='Nao mostrar janelas')

    args = parser.parse_args()

    # Verificar argumentos
    if not any([args.image, args.folder, args.video, args.camera is not None, args.benchmark]):
        parser.print_help()
        print("\nErro: Especifique --image, --folder, --video, --camera ou --benchmark")
        sys.exit(1)

    # Inicializar detector
    try:
        detector = NeedleDetector(model_path=args.model, device=args.device)
    except FileNotFoundError as e:
        print(f"Erro: {e}")
        sys.exit(1)

    # Executar modo selecionado
    if args.benchmark:
        benchmark(detector)
    elif args.image:
        process_image(detector, args.image, args.output, show=not args.no_display)
    elif args.folder:
        process_folder(detector, args.folder, args.output)
    elif args.video:
        process_video(detector, args.video, args.output)
    elif args.camera is not None:
        process_camera(detector, args.camera)


if __name__ == "__main__":
    main()
