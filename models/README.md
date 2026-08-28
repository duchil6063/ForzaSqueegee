# 모델

신경망 모델은 **저장소에 없다** (합쳐 332MB). 목록·크기·SHA-256이
저장소 뿌리의 `release.json`에 있고, `forzasqueegee/modelstore.py`가 **쓰기 직전에** 이 저장소
릴리스에서 받아 이 폴더에 둔다 — 두 번째 실행부터는 네트워크를 안 탄다.

    python -m forzasqueegee models            # 실행에 쓰는 것 넷을 미리 받는다
    python -m forzasqueegee models --check    # 무엇이 있고 없는지만 본다
    python -m forzasqueegee models --verify   # 받아 둔 파일 SHA-256 대조

받는 곳은 `release.json`의 `repo`+`tag`이고, `FS_MODEL_BASE_URL`로 갈아탈 수
있다 (미러를 물릴 자리). 두는 자리는 `FS_MODEL_DIR`로 옮긴다.
`FS_NO_MODEL_FETCH=1`이면 아예 안 받는다 — 그래도 프로그램은 돈다
(아래 각 모델의 "없으면" 항목).

## 릴리스에 올리는 법 (관리자)

`release.json`의 `tag`가 가리키는 릴리스에 파일 이름 **그대로** 올린다.
그 릴리스에는 FLS 편집기 빌드본과 그 대응 소스도 함께 올라간다
(`python tools/fls_build.py --package`):

```
gh release create assets-v1 models/*.onnx dist/ForzaLiveryStudio-itasha-*.zip \
    --title "ForzaSqueegee assets v1"
gh release upload assets-v1 models/anilines_basic.onnx --clobber   # 하나만 갈 때
```

모델을 갈면 `release.json`의 `size`·`sha256`을 다시 적고 (`sha256sum`), 판이
달라지면 `tag`를 올린다 — 옛 판을 쓰는 사람이 안 깨지도록 릴리스는 지우지 않는다.

## 무엇이 있나

- `isnet_anime.onnx` — 신경망 배경 제거 (인물 알파).
  [rembg](https://github.com/danielgatis/rembg)가 배포하는 isnet-anime
  (SkyTNT anime-segmentation 계열, IS-Net 구조, 애니 인물화 특화,
  md5 6f184e756bb3bd901c8849220a83e38e). rembg가 내놓는 파일과 **바이트가 같다**
  (이름만 `isnet_anime.onnx`로 앉는다). torch 없이 onnxruntime CPU로 돈다.
  알파 없는 입력의 전처리(`engine/bgremove.py`)가 쓴다 — 파일이 없으면
  경고만 내고 알파 없이 진행한다.
- `anilines_basic.onnx` — 신경망 선화 추출.
  [AniLines-Anime-Lineart-Extractor](https://github.com/zhenglinpan/AniLines-Anime-Lineart-Extractor)
  (zhenglinpan, MIT)의 basic 모델을 ONNX(opset 17, 동적 H·W)로 변환한 것.
  torch 없이 onnxruntime CPU로 돈다 (1200² 기준 ~3초, torch 원본과 오차 <5e-5).
  cel 노선의 선화 추출(`engine/lineart.py`)이 쓴다 — 못 받으면 고전
  기법으로 자동 대체되고 경고만 낸다. **line 노선은 이것이 필수다.**
- `anilines_detail.onnx` — 같은 저장소의 **detail 판** (69MB).
  같은 방식으로 ONNX(opset 17, 동적 H·W)로 옮긴 것이고, torch 대조 최대
  오차 4.4e-06이다. 가중치는 업스트림 저장소에 없고
  HuggingFace Space `aidenpan/AniLines-Anime-Lineart-Extractor`의
  `weights/detail.pth`(md5 2aca7e307852ece51537fc4306d0c321)에 있다.
  **그 Space도 MIT를 명시한다** (카드 메타데이터 `license: mit` — 2026-08-28
  확인, `weights/basic.pth`와 같은 자리에서 배포된다). 저장소 밖에서 온
  가중치라 출처를 여기에 못 박아 둔다.
  **basic과 입력이 다르다**: basic은 선명화 ×6 한 RGB 3채널, detail은
  회색조와 뒤집은 Sobel 크기 2채널이다 (선명화 없음). 선 재구성이
  **낮은 우선순위 증거**로 쓴다 — 못 받으면 basic만으로 돈다.
- `realesrgan_anime6b.onnx` — 저해상 입력 확대 (애니 특화 SR).
  [Real-ESRGAN](https://github.com/xinntao/Real-ESRGAN) (xinntao, BSD-3-Clause)의
  `RealESRGAN_x4plus_anime_6B`(md5 d58ce384064ec1591c2ea7b79dbf47ba)를
  ONNX로 변환한 것 (RRDBNet 6블록, ×4).
  **원본이 작업 해상도보다 작을 때만** `engine/upscale.py`가 쓴다 — 파일이
  없으면 경고만 내고 큐빅 확대로 진행한다. `FS_NO_SR=1`로 끈다.
