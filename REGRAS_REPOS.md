# REGRAS PARA EVITAR CONFUSAO ENTRE REPOS

## OS 2 PROJETOS

| Projeto | Pasta | GitHub | Funcao |
|---------|-------|--------|--------|
| TRAINER | `~/ultrasound-needle-trainer` | petroscarvslho/ultrasound-needle-trainer | Treina modelos |
| APP | `~/aplicativo-usg-final` | petroscarvslho/aplicativo-usg-final | Usa modelos |

## REGRA 1: IDENTIFICAR O TERMINAL

Sempre que abrir um terminal, rode:

```bash
# Para TRAINER:
cd ~/ultrasound-needle-trainer && export PS1="🏋️ TRAINER $ "

# Para APP:
cd ~/aplicativo-usg-final && export PS1="📱 APP $ "
```

## REGRA 2: NUNCA VERSIONAR ARQUIVOS GRANDES

Arquivos que NUNCA devem ir pro git:
- `*.pt` (modelos PyTorch)
- `*.pth` (checkpoints)
- `*.npy` (arrays NumPy)
- `*.h5` (modelos Keras)
- Pastas: `processed/`, `raw/`, `exports/`, `checkpoints/`

Sempre verifique o `.gitignore` antes de commitar!

## REGRA 3: FLUXO DE TRABALHO CORRETO

```
┌─────────────────────┐
│      TRAINER        │
│                     │
│  1. Baixar dados    │
│  2. Processar       │
│  3. Treinar         │
│  4. Exportar .pt    │
└─────────┬───────────┘
          │
          │  cp models/*.pt ~/aplicativo-usg-final/models/
          ▼
┌─────────────────────┐
│        APP          │
│                     │
│  5. Usar modelo     │
│  6. Rodar app       │
└─────────────────────┘
```

## REGRA 4: ANTES DE COMMITAR

```bash
# SEMPRE verificar ONDE voce esta:
pwd
git remote -v

# SEMPRE verificar O QUE vai commitar:
git status
git diff --stat

# Se tiver arquivo grande (.pt, .npy), NAO commite!
```

## REGRA 5: COMMITS SEPARADOS

- Commits no TRAINER: falam de treino, datasets, modelos
- Commits no APP: falam de interface, plugins, funcionalidades

Se o commit parece "fora do lugar", PARE e verifique se esta no repo certo!

## REGRA 6: UM TERMINAL POR PROJETO

- Terminal 1: SOMENTE para TRAINER
- Terminal 2: SOMENTE para APP

NAO misture! Se precisar copiar arquivo, abra o Finder ou use `cp` com caminho absoluto.

## COMANDOS UTEIS

```bash
# Ver em qual repo estou:
git remote -v

# Ver ultimos commits (conferir se faz sentido):
git log --oneline -5

# Ver arquivos modificados:
git status

# Desfazer mudancas nao commitadas:
git checkout -- .

# Ver tamanho dos arquivos antes de commitar:
git status --porcelain | awk '{print $2}' | xargs ls -lh 2>/dev/null
```

## SE COMMITAR NO REPO ERRADO

```bash
# Desfazer ultimo commit (mantem arquivos):
git reset --soft HEAD~1

# Mover arquivos para repo correto manualmente
# Depois commitar no repo certo
```

---

**Lembre-se**: Os repos sao INDEPENDENTES. Um NAO afeta o outro!
