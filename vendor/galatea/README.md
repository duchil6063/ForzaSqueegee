# vendor/galatea — painter 노선의 KFPS 동봉본

painter 노선(`forzasqueegee/engine/galatea/`)이 실행하는 GPU 원시 생성기와
KFPS 생성 프리셋의 **무수정 사본**이다. 라이선스 전문은
`../../THIRD_PARTY_NOTICES.md`와 이 폴더의 LICENSE 파일 둘에 있다.

## 출처

- 저장소: https://github.com/heyitshestia/kloudys-forza-painter-suite (MIT)
- 가져온 시점: 2026-08-25, 커밋 `0af4f21f984ad42f33dcf570ad36ad8e704092b6`
  (KFPS 버전 3.1.40, 2026-08-24)

## 파일

| 파일 | 무엇 | 출처·검증 |
|---|---|---|
| `KloudysGalateaGenesis.exe` | GPU 원시 생성기 — forza-painter-geometrize-go(Go, OpenCL/Vulkan)의 KFPS 배포 빌드. 실행 배너 `v3.0.0-Genesis-20260714` | SHA-256 `40f362aa4fafcef3f9985a064b3bc32accd8ff46ed330dcecb2d2b41f1444f11` |
| `settings/a.flat-colors.ini` | 평면색 프리셋 (스티커·로고) | KFPS `settings/` 원본 그대로 |
| `settings/b.shaded-art.ini` | 음영 캐릭터 프리셋 (기본) | 〃 |
| `settings/c.gradients.ini` | 그라데이션 프리셋 | 〃 |
| `LICENSE.kfps` | KFPS 저장소 LICENSE 사본 (MIT — forza-painter·geometrize-lib·Primitive 파생 고지) | 〃 |
| `LICENSE.geometrize-gpu` | GPU 생성기 라이선스 사본 (MIT, © 2026 神龟) | KFPS `LICENSE.geometrize-gpu` 원본 그대로 |

## 갱신하는 법

KFPS 저장소를 받아 위 파일들을 그대로 덮고, 이 표의 커밋·버전·SHA-256을
다시 적는다. exe 이름이 바뀌면(`KloudysGeneratorV8.exe` 등)
`forzasqueegee/engine/galatea/base.py`의 `GENERATOR_BIN`도 맞춘다.
프리셋 키가 바뀌면 `settings.write_v2_settings`의 `ordered_keys`는 그대로
둬도 된다 — 모르는 키는 뒤에 그대로 실린다.
