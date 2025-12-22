# INSTRUCOES PARA CONTINUAR EM NOVO TERMINAL (CLAUDE CODE)

====================================================================
## PROJETO RELACIONADO - APLICATIVO USG PRINCIPAL
====================================================================

Este projeto (ultrasound-needle-trainer) e um **sistema de treinamento**
que gera modelos para o plugin NEEDLE PILOT do aplicativo principal.

### PROJETO PRINCIPAL:
- **GitHub**: https://github.com/petroscarvslho/aplicativo-usg-final
- **Local**: /Users/priscoleao/aplicativo-usg-final
- **Plugin**: NEEDLE PILOT v3.1 PREMIUM (src/ai_processor.py)

### ARQUITETURA DOS PROJETOS:
```
┌─────────────────────────────────────┐
│  ultrasound-needle-trainer          │
│  (Projeto de Treinamento)           │
│                                     │
│  - download_datasets.py             │
│  - train_vasst.py                   │
│  - models/vasst_needle.pt  ─────────┼──┐
└─────────────────────────────────────┘  │
                                         │ COPIAR
┌─────────────────────────────────────┐  │
│  aplicativo-usg-final               │  │
│  (Projeto Principal)                │  │
│                                     │  │
│  - main.py                          │  │
│  - src/ai_processor.py              │  │
│    ├── NEEDLE PILOT v3.1 ◄──────────┼──┘
│    ├── NERVE TRACK                  │
│    ├── CARDIAC AI                   │
│    └── ... (10 plugins)             │
│  - models/vasst_needle.pt           │
└─────────────────────────────────────┘
```

### FLUXO DE TRABALHO:
```
1. Baixar datasets (download_datasets.py)
2. Treinar CNN VASST (train_vasst.py)
3. Gerar models/vasst_needle.pt
4. Copiar modelo para aplicativo-usg-final
5. NEEDLE PILOT usa o modelo para detectar agulhas em tempo real
```

### APOS TREINAR, COPIAR MODELO:
```bash
cp /Users/priscoleao/ultrasound-needle-trainer/models/vasst_needle.pt \
   /Users/priscoleao/aplicativo-usg-final/models/
```

====================================================================

## PROMPT PARA COPIAR NO NOVO CHAT

```
Estou continuando o desenvolvimento de um projeto de software.

LOCALIZACAO DO PROJETO: /Users/priscoleao/ultrasound-needle-trainer
REPOSITORIO GIT: https://github.com/petroscarvslho/ultrasound-needle-trainer

AO INICIAR, VOCE DEVE:

1. Ler os arquivos de documentacao:
   - README.md
   - INSTRUCOES_CONTINUIDADE.md

2. Verificar estado do repositorio:
   - git status
   - git log --oneline -5

3. Continuar com as tarefas pendentes listadas abaixo.
```

## ESTRUTURA DO PROJETO

```
ultrasound-needle-trainer/
├── download_datasets.py   # Download e geracao de datasets
├── train_vasst.py         # Treinamento da CNN VASST
├── requirements.txt       # Dependencias Python
├── README.md              # Documentacao principal
├── LICENSE                # Licenca MIT
├── .gitignore             # Arquivos ignorados
│
├── synthetic_needle/      # Dataset sintetico (gerado)
├── kaggle_nerve/          # Dataset Kaggle (baixado)
├── camus_cardiac/         # Dataset CAMUS (baixado)
├── brachial_plexus/       # Dataset Brachial (manual)
├── processed/             # Dados processados para treino
└── models/                # Modelos treinados (.pt)
```

## COMO USAR

### 1. Instalar dependencias
```bash
pip install -r requirements.txt
```

### 2. Gerar dataset sintetico
```bash
python download_datasets.py
# Opcao 5 -> Sintetico
# 5000 amostras
```

### 3. Treinar modelo
```bash
python train_vasst.py
# Opcao 1 -> Treinar
```

### 4. Modelo salvo em
```
models/vasst_needle.pt
```

## DATASETS SUPORTADOS

| # | Dataset | Imagens | Acesso |
|---|---------|---------|--------|
| 1 | Kaggle Nerve | 5,635 | API publica |
| 2 | CAMUS Cardiac | 4,000+ | Registro gratis |
| 3 | Brachial Plexus | 41,000 | Busca academica |
| 4 | Breast US | 780 | Kaggle API |
| 5 | Sintetico | Ilimitado | Geracao local |

## LINKS UTEIS

- Kaggle: https://www.kaggle.com/c/ultrasound-nerve-segmentation
- CAMUS: https://www.creatis.insa-lyon.fr/Challenge/camus/
- IEEE DataPort: https://ieee-dataport.org
- Zenodo: https://zenodo.org/search?q=ultrasound
- Papers With Code: https://paperswithcode.com/datasets?q=ultrasound+needle

## DATASETS BAIXADOS

### Brachial Plexus (REAL - com agulha!)
- **Local**: `/Users/priscoleao/ultrasound-needle-trainer/brachial_plexus/`
- **Arquivos**: 42,321 frames
- **Maquinas**: Butterfly, Sonosite, eSaote
- **Anotacoes**:
  - Nervos: `data/<machine>/bb_annotations/`
  - Mascaras: `data/<machine>/ac_masks/`
  - **AGULHA**: `data/Sonosite/needle/needle_coordinates/`

### Proximo Passo: Processar e Treinar com Dados Reais
```bash
cd /Users/priscoleao/ultrasound-needle-trainer
source venv/bin/activate
# Criar script de processamento do brachial_plexus
# Treinar modelo com dados reais
```

---

## ULTIMO TREINAMENTO REALIZADO (Sintetico)

### Data: 2025-12-22

### Configuracao:
- **Dataset**: Sintetico (5000 amostras)
- **Treino/Validacao**: 4250/750 amostras
- **Epocas**: 77 (early stopping)
- **Batch size**: 32
- **Learning rate**: 0.0001
- **Device**: MPS (Apple Silicon GPU)

### Resultados:
- **Melhor Val Loss**: 0.012281
- **Erro medio**: ~3.1 pixels
- **Modelo salvo**: models/vasst_needle.pt (151 MB)
- **Grafico**: models/training_plot.png

### Observacoes:
- Early stopping ativado apos 77 epocas (patience=15)
- Modelo copiado para aplicativo-usg-final/models/
- Fix aplicado: removido parametro `verbose=True` do ReduceLROnPlateau (deprecado no PyTorch recente)

## ESTADO ATUAL

### Ultima atualizacao: 2025-12-22

### Concluido:
- [x] Script de download de datasets (download_datasets.py)
- [x] Script de treinamento CNN (train_vasst.py)
- [x] Geracao de dataset sintetico
- [x] Suporte a Kaggle, CAMUS, Brachial Plexus
- [x] Arquitetura VASST em PyTorch
- [x] Data augmentation
- [x] Early stopping
- [x] Learning rate scheduler
- [x] Visualizacao de predicoes
- [x] Primeiro treinamento completo (~3.1px erro)
- [x] Modelo integrado ao aplicativo-usg-final
- [x] Script de inferencia (inference.py)
- [x] Modulo de metricas (metrics.py)
- [x] Cross-validation (cross_validation.py)
- [x] Benchmark (benchmark.py)
- [x] Docker (Dockerfile, docker-compose.yml)
- [x] Testes unitarios (tests/ - 39 testes)
- [x] Download dataset Brachial Plexus (42,321 frames REAIS)
- [x] Script process_brachial.py para processar Sonosite com anotacoes de agulha
- [x] Script export_vasst.py para state_dict compatível com inferencia
- [x] Script sync_unified_exports.py para usar datasets unificados

---

====================================================================
## PROXIMOS PASSOS (PASSO A PASSO)
====================================================================

### OBJETIVO: Treinar com Dados REAIS (Brachial Plexus)

O dataset ja foi baixado em:
`/Users/priscoleao/ultrasound-needle-trainer/brachial_plexus/`

### PASSO 1: Entender a Estrutura do Dataset
```bash
cd /Users/priscoleao/ultrasound-needle-trainer
source venv/bin/activate

# Ver estrutura
ls -la brachial_plexus/data/
# Resultado esperado: Butterfly/, Sonosite/, eSaote/

# Anotacoes de AGULHA (so no Sonosite):
ls brachial_plexus/data/Sonosite/needle/needle_coordinates/
```

### PASSO 2: Processar Dataset Real (Sonosite)
Rodar o script `process_brachial.py` para gerar os arquivos:
`processed/brachial_real/images.npy` e `processed/brachial_real/labels.npy`
e tambem os splits `processed/X_train.npy`, `Y_train.npy`, etc.

```bash
python process_brachial.py

# Opcional: nao gerar splits
python process_brachial.py --no-splits
```

### PASSO 3 (Opcional): Sincronizar Datasets Unificados
Caso esteja usando o `unified_dataset_manager.py` do app principal:

```bash
python sync_unified_exports.py \
  --source /Users/priscoleao/aplicativo-usg-final/datasets/unified/exports/needle \
  --dest processed
```

### PASSO 4: Treinar com Dados Reais
```bash
python train_vasst.py
# Escolher opcao para treinar com brachial_plexus
```

### PASSO 5: Exportar Pesos Compatíveis com Inferencia
```bash
python export_vasst.py --checkpoint models/vasst_needle.pt --output models/vasst_needle.pt
```

### PASSO 6: Comparar Resultados
- Modelo sintetico: ~3.1px erro
- Modelo real: ???

### PASSO 7: Copiar Modelo Final
```bash
cp models/vasst_needle.pt /Users/priscoleao/aplicativo-usg-final/models/
```

---

====================================================================
## ARQUITETURA HIBRIDA PARA TODOS OS PLUGINS (FUTURO)
====================================================================

Usuario escolheu abordagem HIBRIDA:
- Datasets centralizados em UM local
- Scripts de treinamento SEPARADOS por plugin

### Estrutura Proposta:
```
/Users/priscoleao/
├── aplicativo-usg-final/           # App principal
│   └── datasets/                   # Datasets centralizados
│       ├── unified_dataset_manager.py  # Ja existe!
│       ├── needle/                 # NEEDLE PILOT
│       ├── nerve/                  # NERVE TRACK
│       ├── cardiac/                # CARDIAC AI
│       └── ...
│
└── ultrasound-needle-trainer/      # Treinamento NEEDLE
    ├── train_vasst.py              # Script especifico
    └── models/                     # Modelos gerados
```

### Vantagens do Hibrido:
1. Dados organizados em um so lugar
2. Cada plugin tem seu script otimizado
3. Facil reusar dados entre plugins
4. Nao mistura codigos diferentes

---

## PROXIMAS MELHORIAS (BACKLOG):
- [ ] **PRIORIDADE**: Processar brachial_plexus e treinar com dados reais
- [ ] Baixar mais datasets (Kaggle Nerve, CAMUS)
- [ ] Implementar arquitetura hibrida completa
- [ ] Transfer learning entre plugins
- [ ] Interface web para visualizar treinamento
