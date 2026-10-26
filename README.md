# 검증형 Manim 시각화 생성기

> 대한전자공학회 「전자공학회지」 2026년 특집 기고문 「바이브 코딩으로 상상하고 수학 애니메이션 도구로 그리는 공학」(정태원·안성수, 게재 예정)의 구현 코드입니다. 논문의 수치를 다시 얻는 방법은 아래 [논문 결과 재현](#논문-결과-재현)을 보세요.

슬라이더로 값을 바꾸고, **검증을 통과한 경우에만** Manim 영상을 만드는 웹 화면입니다.

## 구성

| 파일 | 역할 |
|---|---|
| `app.py` | 웹 화면 (Gradio) |
| `llm_router.py` | 자연어 요청 → 로컬 LLM(Ollama) → 검증된 템플릿 설정 |
| `backprop_core.py` / `backprop_demo.py` | 역전파: 계산·검증 / 렌더링 |
| `attention_core.py` / `attention_demo.py` | 셀프 어텐션: 계산·검증 / 렌더링 |
| `gcn_core.py` / `gcn_demo.py` | GCN 메시지 패싱: 계산·검증 / 렌더링 |
| `gan_core.py` / `gan_demo.py` | GAN 모드 붕괴: 실제 학습·검증 / 렌더링 |
| `layout_check.py` | 화면 배치 자동 검사 (겹침·이탈·관통) |

## 설치와 실행

필요한 것: Python 3.10 이상, [Manim Community](https://docs.manim.community/) 실행 환경(FFmpeg, 수식용 LaTeX), 한글 글꼴(기본 맑은 고딕, 아래 '참고' 참조). 자연어 요청 탭을 쓰려면 [Ollama](https://ollama.com/)와 로컬 모델이 추가로 필요합니다.

```bash
git clone https://github.com/onomcanbot/manim-verified-visualizer.git
cd manim-verified-visualizer
pip install -r requirements.txt
python app.py
```

conda를 쓰는 경우 Manim이 설치된 가상환경을 먼저 활성화(`conda activate <환경 이름>`)한 뒤 같은 순서로 실행합니다.

브라우저가 자동으로 열립니다. 열리지 않으면 http://127.0.0.1:7860 에 접속하세요.

## 자연어로 요청 (로컬 Ollama)

Ollama를 켜 둔 상태에서 `python app.py` → 첫 번째 탭 **자연어로 요청 (Ollama)**

1. **모델 목록 불러오기** (화면이 열릴 때 자동으로 한 번 시도)
2. 요청 입력 → **① 해석하기** : LLM이 템플릿과 설정을 JSON으로 고름 → 형식·범위 검사 → 계산 모듈 검증
   (실패하면 오류를 LLM에 되돌려 최대 3번 자동 수정)
3. **② 영상 만들기**

LLM은 코드를 쓰지 않습니다. LLM이 쓴 설명은 참고용이며, 영상의 숫자와 문장은 계산 결과에서 만들어집니다.

명령줄:

```cmd
python llm_router.py --models
python llm_router.py "정답을 0으로 바꾼 역전파 보여줘" --model gemma4:26b
python llm_router.py --bench --model gemma4:26b          # 벤치마크(개발 18 + 보류 20문항) → web_runs/bench_*.json
python llm_router.py --bench --model gemma4:26b,hermes3:8b   # 여러 모델 연속 실행 + 비교표
```

- 문항 세트: `--set dev` (개발 18문항, 프롬프트를 만들 때 참고) · `--set holdout` (보류 20문항, 프롬프트를 완성한 뒤 따로 작성) · 기본은 둘 다
- 프롬프트: 기본값이 논문에 쓴 **최종 프롬프트**입니다. 개발 초기 기준선(`--prompt v1`)은 비교용으로 코드에만 남겨 두었고 논문에는 쓰지 않았습니다.
- 결과가 개발 문항에만 맞춰진 것인지는 **보류 세트** 결과로 판단합니다
- LLM 호출은 온도 0, 시드 42로 고정합니다

벤치마크 지표:

- **라우팅** : 템플릿을 맞게 고르고 검증을 통과 (지원하지 않는 요청은 거절)
- **의도 일치** : 라우팅 + 실제 쓰일 설정이 요청의 구체적 값과 일치 (예: "두 번째 단어" → focus = 1)
  검증기는 계산이 맞는지를 보장하지만 요청과 같은지는 보장하지 못하므로 따로 측정합니다
- **첫 시도 성공 · 평균 수정 횟수 · 평균 소요 시간**
- 틀린 문항은 시도별 LLM 원문과 문제가 함께 출력되고, 전체 기록은 `web_runs/bench_*.json`에 저장됩니다

모델 고르기:

- 기본값은 설치된 **로컬** 모델 중 `gemma4:26b` (없으면 다른 gemma4·qwen 계열)
- 이름에 `cloud`가 붙은 모델(예: `gemma4:31b-cloud`)은 요청이 PC 밖 서버로 전송됩니다. 화면에 따로 표시되며 기본값으로는 고르지 않습니다
- Ollama 앱 목록에서 다운로드 아이콘(↓)이 붙은 모델은 아직 설치되지 않은 것입니다. `ollama pull <모델>`로 받은 뒤 사용하세요

연결이 안 되면: `ollama list`로 Ollama가 켜져 있는지, 주소·포트(기본 11434)가 맞는지 확인하세요.

## 사용 순서 (각 템플릿 탭)

1. 왼쪽에서 값을 바꾼다
2. **① 검증하기** — 통과/실패와 사유를 표시
3. **② 영상 만들기** — 검증을 통과해야만 렌더링 시작

## 검증 관문 (세 곳)

1. 웹 화면: 영상 만들기를 눌러도 검증을 다시 실행
2. 서버: 검증 실패 시 manim 자체를 실행하지 않음
3. Manim 씬: 씬 내부에서도 재검증, 실패하면 렌더링 중단

## 결과 저장 (재현성)

실행할 때마다 `web_runs/<데모>_<날짜_시각>/` 폴더에 `config.json`(설정), 영상, `render.log`가 함께 저장됩니다.
같은 설정 파일로 언제든 같은 영상을 다시 만들 수 있습니다.

```cmd
set BACKPROP_CONFIG=web_runs\backprop_20261007_120000\config.json
manim -pql backprop_demo.py BackpropChainRule
```

## 명령줄에서 검증만

```cmd
python backprop_core.py --inject
python attention_core.py --inject
python gcn_core.py --inject
python gan_core.py --inject --seeds 5     # 학습 포함, 2~3분
```

## 화면 배치 자동 검사 (글자 겹침 · 화면 이탈 · 선이 글자를 관통)

렌더링 없이 장면을 끝까지 실행하면서 단계마다 글자 위치를 재서 문제를 보고합니다. 새 장면을 만들거나 문구를 바꾼 뒤 돌려 보세요.

```cmd
python layout_check.py backprop_demo.py BackpropChainRule
python layout_check.py gcn_demo.py GCNMessagePassing
set SNAP_STEPS=51 && python layout_check.py backprop_demo.py BackpropChainRule   # 51단계 정지 화면 저장
```

3D 장면은 화면 고정 글자만 검사하므로 실제 렌더링 화면으로 한 번 더 확인하세요.

## 논문 결과 재현

이 저장소를 소개한 전자공학회지 특집원고의 수치는 아래 명령으로 다시 얻을 수 있습니다. 계산 모듈(`*_core.py`)은 NumPy·SciPy만 쓰므로 Manim 없이도 실행됩니다.

| 본문 | 명령 | 나와야 하는 결과 |
|---|---|---|
| Ⅲ.2 역전파 수치미분 대조 | `python backprop_core.py` | 데모 값 최대 오차 3.98e-12(≈4.0×10⁻¹²), 허용 1e-8 → PASS |
| Ⅳ.1 역전파 1회 갱신 | (같은 명령) | 손실 0.0829 → 0.0658 |
| Ⅳ.1 셀프 어텐션 | `python attention_core.py` | '먹었다' → '생선을' 가중치 0.52, 독립 구현 대조 PASS |
| Ⅳ.1 GCN | `python gcn_core.py` | 독립 구현·불변식 5종 대조 PASS (노드 C는 자기 자신 포함 메시지 4개) |
| Ⅳ.1 GAN 모드 붕괴 | `python gan_core.py` | 기본 GAN 1/8 모드, PacGAN 8/8 모드 (3,000단계, 수 분 소요) |
| Ⅳ.1 시드 0~4 요약 | `python gan_core.py --seeds 5` | 덮은 모드 평균: 기본 GAN 1.4, PacGAN 7.4 |
| Ⅳ.2 표 3 오류 주입 18종 | `python backprop_core.py --inject` · `attention_core.py --inject` · `gcn_core.py --inject` · `gan_core.py --inject` | 3/3 · 4/4 · 6/6 · 5/5 차단 (GAN은 600단계 학습으로 시험) |
| Ⅲ.3 깔끔한 예제가 숨기는 버그 | (위 `--inject` 출력) | 어텐션 'V 대신 K'와 GCN 정규 그래프의 행 정규화는 데모 값만으로는 PASS(놓침), 무작위 20세트를 더하면 FAIL |
| Ⅳ.3 표 4 LLM 벤치마크 | `python llm_router.py --bench --model gemma4:26b,hermes3:8b` | Ollama와 두 모델 필요. 라우팅 95% 이상, 의도 일치 89~100% (RTX 3090 기준, 모델 판에 따라 달라질 수 있음) |
| Ⅳ.4 화면 배치 검사 | `python layout_check.py backprop_demo.py BackpropChainRule` (나머지: `attention_demo.py SelfAttention`, `gcn_demo.py GCNMessagePassing`, `gan_demo.py GANModeCollapse`) | 네 장면 모두 문제 0건 (Manim 필요) |

웹 화면의 역전파 탭은 현재 설정의 오차와 무작위 20세트를 포함한 최악 오차를 함께 보여 줍니다. 본문의 4.0×10⁻¹²는 앞의 값입니다.

## 참고

- 한글 폰트 기본값은 맑은 고딕. 바꾸려면 `set KO_FONT=NanumGothic`
- 어텐션 영상 레이아웃은 단어 3개, 2차원 임베딩 기준
- GCN 영상 레이아웃은 노드 5개, 2차원 특징 기준 (간선은 자유롭게)
- GAN 탭은 검증 단계에서 실제 학습과 재학습을 하므로 검증에만 수십 초가 걸립니다

## 라이선스

MIT License. 자세한 내용은 [LICENSE](LICENSE)를 보세요.
