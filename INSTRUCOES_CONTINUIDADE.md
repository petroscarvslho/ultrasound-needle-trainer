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

### FLUXO DE TRABALHO:
```
[ultrasound-needle-trainer]          [aplicativo-usg-final]
         |                                    |
   1. Baixar datasets                         |
   2. Treinar CNN VASST                       |
   3. Gerar vasst_needle.pt                   |
         |                                    |
         +-------> COPIAR MODELO ------------>+
                                              |
                                    4. NEEDLE PILOT usa o modelo
                                    5. Detecta agulhas em tempo real
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

### Proximas melhorias:
- [ ] Adicionar suporte a mais datasets
- [ ] Implementar cross-validation
- [ ] Adicionar metricas (precision, recall)
- [ ] Criar script de inferencia separado
- [ ] Adicionar testes unitarios
- [ ] Docker container para reproducibilidade
