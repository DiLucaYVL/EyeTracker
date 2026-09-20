# EyeMouse

Mouse controlado pelo **olhar** (webcam comum), com **cliques por pinça de mão**. Funciona 100% local, sem internet
depois da primeira execução (que baixa dois modelos do MediaPipe).

- **Olhar → cursor**: o cursor segue para onde você olha; uma *bolha* translúcida mostra o ponto estimado (como no GazeRecorder).
- **Clique esquerdo**: pinça do **polegar com o indicador**. **Clique direito**: pinça do **polegar com o médio**.
  Segure a pinça por mais de ~0,35 s para **arrastar**.
- **Calibração em tela cheia** que aprende: cada calibração, teste ou clique de correção **acumula** amostras.
- **Suporte a movimento da cabeça**: a calibração tem uma 2ª fase em que você mexe a cabeça enquanto olha os pontos.
- **Tela de imagem**: ajustes de brilho/contraste/gama/CLAHE/nitidez e o painel da câmera, para o olho ficar bem contrastado.

![Posição da cabeça em cada etapa da calibração](docs/guia_posicao_da_cabeca.png)

## Instalador (Windows)

Para quem só quer usar: rode `EyeMouse-Setup-<versão>.exe` (não precisa de Python). Ele instala por usuário (sem pedir
administrador), cria o atalho no Menu Iniciar, já traz os modelos do MediaPipe (funciona offline) e tem desinstalador.
Sua calibração e configurações ficam em `%APPDATA%\EyeMouse` e só são apagadas se você aceitar na desinstalação.

- O instalador não é assinado digitalmente: o SmartScreen pode avisar. Use *Mais informações → Executar assim mesmo*.
- `EyeMouse-console.exe` (na pasta instalada) é o mesmo programa com console: `EyeMouse-console.exe --doctor` diagnostica
  câmera e detecção. Erros do app instalado são gravados em `%APPDATA%\EyeMouse\eyemouse.log`.

**Gerar o instalador** (o `.exe` não é versionado no git, pasta `dist/`):

```bat
.venv\Scripts\pip install -r requirements-build.txt
winget install JRSoftware.InnoSetup
build_installer.bat
```

Isso roda o PyInstaller (`EyeMouse.spec` → `build\dist\EyeMouse`, ~300 MB) e depois o Inno Setup (`installer\eyemouse.iss`
→ `dist\EyeMouse-Setup-<versão>.exe`, ~95 MB). `build_installer.bat --skip-installer` gera só a pasta do aplicativo.
A versão vem de `eyemouse/__init__.py`.

## Instalar e executar (desenvolvimento)

```bat
python -m venv .venv
.venv\Scripts\pip install -r requirements.txt
run.bat
```

Na primeira execução são baixados `face_landmarker.task` (~4 MB) e `hand_landmarker.task` (~8 MB) para `data/`.
Requer Windows e Python 3.10+ (testado no 3.13).

Comandos úteis:

| Comando | O que faz |
|---|---|
| `run.bat` | abre o painel; se não houver calibração, abre a calibração automaticamente |
| `run.bat --calibrate` | abre a calibração ao iniciar |
| `run.bat --doctor` | testa câmera, modelos e taxa de detecção de rosto/mão (sem interface) |
| `run.bat --reset` | apaga a calibração salva |
| `run.bat --camera 1` | usa outra câmera |

## Como usar (passo a passo)

1. **Imagem** (botão *Imagem e contraste dos olhos*, ou tecla `I` na calibração). Em tela cheia você vê a câmera, uma faixa
   ampliada dos olhos e um veredito ("Olhos escuros demais", "Bom contraste nos olhos ✓"…).
   - Aba **Software**: brilho, contraste, gama, CLAHE, nitidez — aplicados antes da detecção. `A` = auto-ajustar pelos olhos.
   - Aba **Câmera (driver)**: exposição, ganho, brilho, contraste, saturação, gama, nitidez, luz de fundo (lidos e gravados
     direto na câmera). `P` abre o painel nativo do Windows ("Video Proc Amp").
   - `O` alterna entre imagem original e ajustada para comparar.
2. **Calibração** (`Ctrl+Alt+C`). Tela cheia, com o modelo 3D da cabeça mostrando a posição ideal (ciano) e a sua cabeça ao vivo (branco):
   - *Fase 1*: 9/16/25 pontos com a **cabeça parada**.
   - *Fase 2* (`H` liga/desliga): 9 pontos com a **cabeça em movimento** (lados, cima/baixo, perto/longe, círculo).
     O quadrado "poses da cabeça" mostra quais poses já foram capturadas.
   - *Refinamento*: a bolha aparece ao vivo. Olhe para um ponto e **clique nele** com o mouse: vira uma amostra e o modelo
     se ajusta na hora. `T` roda um **teste guiado** (8 pontos aleatórios) que mostra o erro e já aprende com eles.
3. **Ligue o mouse**: botão *Mouse com os olhos* ou `Ctrl+Alt+E`. Ele **sempre inicia desligado**, por segurança.
4. Com o mouse desligado, cada clique físico seu (quando a bolha estava perto) também ensina o modelo
   (*Aprender com cliques do mouse*).

Atalhos globais: `Ctrl+Alt+E` liga/desliga o mouse • `Ctrl+Alt+B` mostra/oculta a bolha • `Ctrl+Alt+C` calibrar • `Ctrl+Alt+H` modo da cabeça • `Ctrl+Alt+M` modo da mão • `Ctrl+Alt+R` recentralizar a cabeça • `Ctrl+Alt+Q` sair.

## Modos de controle (painel principal)

Dois menus no painel escolhem **o que move o mouse**. O modo é salvo e vale na próxima abertura.

| Menu | Modo | O que faz |
|---|---|---|
| **Cabeça** | Olho *(padrão)* | O cursor segue o olhar. A pose da cabeça continua entrando no modelo para **compensar** o erro, mas mexer a cabeça **não move** o mouse. Precisa de calibração. |
| | Cabeça + olho | O olhar posiciona o cursor e virar a cabeça o desloca também (cabeça = movimento grosso, olho = ajuste fino). Precisa de calibração. |
| | Cabeça | O cursor segue para onde o nariz aponta, relativo a uma pose neutra. Não precisa de calibração. |
| | Desativado | Olho e cabeça **não movem** o mouse. Os gestos da mão continuam: com a mão em *Pinça* você usa o mouse físico e clica/rola com a mão; com *Ponta do indicador move o cursor*, o dedo também move. Não precisa de calibração. |
| **Mão** | Pinça (clique) *(padrão)* | A mão só clica e rola: polegar+indicador = esquerdo, polegar+médio = direito, segurar = arrastar, polegar+anelar = rolagem. Vale **qualquer uma das duas mãos**. |
| | Ponta do indicador move o cursor + pinça | Quem move o cursor é **só a ponta do dedo indicador** (a bola do indicador), não a mão inteira; basta a mão estar visível, sem pose obrigatória. A pinça clica **e a ponta continua movendo o cursor durante a pinça** (é assim que se arrasta; o cursor segue o movimento *relativo* a partir do clique, então o clique não vira arrasto sem querer). **Polegar+anelar** rola a página e para o cursor. As duas mãos valem: uma pode apontar enquanto a outra rola. Sem mão, vale o modo de cabeça. |

- **Recentralizar cabeça** (`Ctrl+Alt+R`): define a pose atual como o centro da tela nos modos com cabeça. Isso também acontece ao ligar o mouse e ao trocar de modo.
- **Sensibilidade do movimento** (sliders 0,3–3,0): *Cabeça*, *Mão* e *Rolagem*. O modo Olho não tem sensibilidade: sua precisão vem da calibração.
- Atalhos: `Ctrl+Alt+H` alterna o modo da cabeça, `Ctrl+Alt+M` o da mão.

## Rolagem com polegar+anelar

Encostar a **bola do polegar na bola do dedo anelar** liga o modo de rolagem; enquanto elas se tocam, mover a mão rola a página como dois dedos no
touchpad: quanto mais rápido o movimento, mais intensa a rolagem (curva superlinear; a sensibilidade é o slider *Rolagem*). Vertical e horizontal,
com bloqueio de eixo (só a direção dominante rola). Mão para baixo rola para baixo; *Inverter a direção da rolagem (natural)* troca isso.
Separar os dedos encerra a rolagem. Durante a rolagem o cursor fica parado (no modo "Ponta do indicador move o cursor") e a bolha fica roxa.
Pode ser desligada em *Rolar com polegar+anelar*. Uma mão em rolagem não clica; a outra mão continua livre. Se o polegar estiver entre o
médio/indicador e o anelar, vale o dedo **mais próximo**. Vale qualquer uma das duas mãos.

## Visão da câmera

O botão **Ver câmera (visão computacional)** abre uma janela ao vivo com o que o programa enxerga: rosto (caixa e íris), **esqueleto da mão**, estado
(*relaxada*, *PINÇA esquerda/direita*, *ROLAGEM* com a velocidade, **um estado por mão**), as **bolas** nas pontas dos dedos (polegar, indicador, médio, anelar; ficam preenchidas quando se tocam: verde = clique esquerdo, laranja = direito, roxo = rolagem), a distância das pontas e a origem do cursor.
Use-a para ajustar a posição da mão: com a mão fora do quadro nada funciona (foi a causa de testes ruins). É a mesma janela usada nos testes ao vivo.

## Dicas para acertar mais

- Luz **na frente** do rosto (janela atrás de você deixa o rosto em silhueta e o rastreio falha).
- Com pouca luz a webcam alonga a exposição e cai para ~10–15 fps. Acenda uma luz ou reduza a *Exposição* na aba Câmera e compense com Ganho/Gama/CLAHE.
- Câmera na altura dos olhos, ~50 cm da tela. Se mudar de posição, refaça (ou refine) a calibração.
- Precisão realista de webcam: erro típico de 2–5% da tela. Não substitui um rastreador infravermelho.
- A câmera precisa **enxergar sua mão** para a pinça funcionar. **O único critério de pinça é: as bolas dos dois dedos se tocam.** Cada ponta
  de dedo (polegar, indicador, médio, anelar) é uma bola desenhada na *Visão da câmera*, com raio = *Bola da pinça* (% do tamanho da mão, padrão 9,5%).
  Duas bolas se tocam quando a distância entre as pontas, medida **na imagem**, é no máximo dois raios. Tocando = pinça, não tocando = sem pinça:
  sem 3D, sem histerese, sem espera e sem "guarda de punho"; o clique é solto no primeiro quadro em que as bolas se separam.
- **Tamanho da bola**: digite no painel principal (*Bola da pinça*, 3–30 % da mão) ou rode o assistente **`P`** na tela inicial da calibração
  (15 s: mede sua mão aberta e em pinça e escolhe a bola que encosta quando *os seus* dedos encostam). Bola maior = dispara com os dedos mais afastados;
  menor = exige encostar de verdade. "Bola da pinça: padrão" está em *Configurações…*.
- Como a única regra é o toque das bolas, um punho fechado também pode aproximar o polegar do indicador. A única arbitragem: uma mão em **rolagem**
  (polegar+anelar) não clica, e o dedo mais próximo do polegar decide entre clique e rolagem.
- Limite conhecido: se o polegar passar rapidamente por cima da ponta do indicador (sem tocar de verdade na profundidade), a câmera única não distingue
  isso de uma pinça muito rápida e pode gerar um clique breve.
- Alterações na aba *Câmera* **persistem no driver**, afetam outros apps e são reaplicadas toda vez que o programa abre. Se a imagem
  ficar escura ou estranha, abra *Imagem e contraste* e aperte **`R` (Restaurar tudo)**: volta o driver ao estado original e zera os ajustes de software.
  O programa avisa quando a imagem está quase preta.

## Configuração

`data/config.json` (gerado automaticamente; quase tudo também está em *Configurações…*). Principais chaves:
`smoothing_min_cutoff` / `smoothing_beta` (suavização), `deadzone_px`, `bubble_size`, `pinch_ball_size`
(tamanho da bola da pinça, fração do tamanho da mão), `drag_hold_ms`, `calibration_points`, `calibration_head`, `img_*` (imagem), `camera_props`, `camera_fourcc`.

## Como funciona

```
câmera ─► ajuste de imagem ─► MediaPipe FaceLandmarker (íris, blendshapes, pose 3D da cabeça)
                            └► MediaPipe HandLandmarker (pinça)
íris/olho + pose da cabeça ─► regressão ridge (grau 1/2/3 escolhido por validação cruzada)
                          ─► mediana + filtro 1€ ─► cursor (SetCursorPos) e bolha
pinça ─► SendInput (botão esquerdo/direito, segurar = arrastar)
```

- **Features** (11): posição da íris em coordenadas do olho (por olho), abertura da pálpebra, yaw/pitch/roll e posição da cabeça.
- **Aprendizado incremental**: o modelo guarda todas as amostras (`data/calibration.json`) e reajusta a cada nova informação;
  cada alvo tem o mesmo peso, e amostras discrepantes são atenuadas (Huber).
- O olho fechado congela o cursor (a estimativa do olhar é inválida enquanto pisca).

Arquivos: `eyemouse/` (`tracker.py` laço principal, `gaze_model.py`, `features.py`, `hands.py`, `calibration_ui.py`,
`image_ui.py`, `head3d.py`, `imaging.py`, `control.py` (modos e rolagem), `preview_ui.py` (visão da câmera), `bubble.py`, `app.py`), `tests/`, `tools/render_head_guide.py` (gera a imagem acima).

## Testes

```bat
.venv\Scripts\python -m unittest discover -s tests -t .
```

## Ideias reaproveitadas de outros projetos

[EyeTrax](https://github.com/ck-zhang/EyeTrax) (calibrações 9/5/densa, filtros, modelo persistente),
[EyeControl](https://github.com/medboughrara/EyeControl-Real-Time-Webcam-Eye-Tracking-Mouse) (FaceLandmarker + blendshapes, ridge polinomial, filtro 1€),
[eye_mouse](https://github.com/NicholasMilani/eye_mouse) (mediana + suavização + zona morta),
[edpl22/eye-mouse](https://github.com/edpl22/eye-mouse) (calibração de 9 pontos com compensação de cabeça).
O código deste repositório foi escrito do zero; só as ideias foram aproveitadas.
