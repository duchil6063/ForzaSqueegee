# Third-Party Notices

ForzaSqueegee 본체는 MIT다 (`LICENSE`) — 다만 `tools/fls-patch/*.patch`는
AGPL-3.0-or-later이고 `catalog/`의 게임 추출 자료는 우리 것이 아니다. 그
경계는 `LICENSE`가 적는다. 그와 별개로 아래 오픈소스 프로젝트에서
유래한 것을 함께 쓴다 — GPU 도형 생성기 실행 파일과 프리셋을 `vendor/galatea/`에,
KFPS Fabric 비닐 편집기를 `vendor/kfps-editor/`에 동봉하고(도형 메시 리소스는
빼고 — 게임 자료라 KFPS 고정 커밋에서 받는다),
신경망 모델(ONNX 변환본)과 **우리 패치를 얹어 지은 ForzaLiveryStudio 빌드본**은
저장소에 넣지 않고 이 프로젝트의 릴리스에서 각각 `models/`·`vendor/fls-editor/`로
**받아서** 쓴다 (자리는 `release.json`).
각 원본 저장소의 저작권 고지와 라이선스 전문을 여기에 재현한다. 모델의
유래·변환 방법·받는 곳은 `models/README.md`, 동봉 실행 파일·편집기의
출처·커밋·해시는 `vendor/galatea/README.md`·`vendor/kfps-editor/README.md`·
`vendor/fls-editor/README.md` 참조.

실행 시 pip로 설치되는 라이브러리(numpy, opencv-python, Pillow, PySide6,
onnxruntime)는 이 배포물에 동봉되지 않으며 각 프로젝트의 라이선스를 따른다.

---

## 게임 콘텐츠 — **오픈소스 라이선스 대상이 아니다**

ForzaSqueegee는 Microsoft Corporation·Playground Games와 **아무 관계가 없는**
비공식 팬 제작 도구다. 그들이 만들지도, 후원하지도, 승인하지도 않았다. Forza,
Forza Horizon은 Microsoft의 상표이고, 차량·제조사 이름은 그 차를 가리키기
위해서만 쓴다.

아래 자료는 게임의 설치 에셋에서 뽑았거나 인게임 실측으로 뜬 것이다. 저작권은
Microsoft·Playground Games를 비롯한 각 권리자에게 있으며, **이 저장소의 MIT
라이선스는 여기에 적용되지 않는다** — 우리는 이 자료에 어떤 권리도 부여하지
않는다. 게임이 읽고 쓰는 파일과 오갈 수 있게 하려고(상호운용) 함께 둘 뿐이고,
권리자의 요청이 있으면 제거한다.

| 자리 | 무엇 | 어떻게 얻었나 |
|---|---|---|
| `catalog/vinyl_catalog.json` | 비닐 도형 1,480종의 정점 루프 | 설치본 `media/Livery/Vinyls.zip` |
| `catalog/shape_native.json` | 도형 native 상자 | 같은 에셋의 BBox 청크 |
| `vendor/kfps-editor/shape-names.json` · `shape-words.json` | 같은 도형의 이름표 | KFPS 동봉본을 거쳐 |
| `catalog/surfaces/` | 차종별 면 마스크·배율 | 인게임 차분 실측 (프로브) |
| `catalog/cell_map.json` · `body_tabs.json` · `font_cells.json` · `paint_parts.json` · `gradient_catalog.json` | 도형 선택 화면의 칸↔도형 대응, 면 탭, 글꼴 칸, 도색 부위, 그라데이션 프로파일 | 인게임 화면 실측 |
| `catalog/fh6_layout.json` | 레이어 레코드 배치 | 인게임 메모리 실측 |

KFPS를 거쳐 온 도형 자료도 마찬가지다 — **KFPS의 MIT는 KFPS가 쓴 것에 대한
것이지, 게임 자료를 재허락할 권한이 아니다.**

**도형 메시도 배포하지 않는다.** 편집기가 그리는
`vendor/kfps-editor/Resources/`(2,800파일 30MB)는 게임 도형의 메시 데이터라
저장소에 넣지 않는다 — 편집기를 처음 열 때 KFPS 고정 커밋에서 받고
(`tools/get_kfps.py`), 받은 것이 그 커밋 그대로인지 집계 SHA-256으로 대조한다.
편집기를 안 쓰면 받지도 않는다.

**차량 색인은 배포하지 않는다.** 면 구성·면 크기 표(`work/state/cars.json`)는
쓰는 사람의 설치본에서 그 자리에서 뜬다 (`game/cars.py` — 처음 필요해질 때
`media/Cars/*.zip`의 `LiveryMasks/Masks.xml`을 읽는다). 판·DLC마다 차가 달라
각자의 설치본이 정본이기도 하고, 그만큼 게임에서 나온 자료를 우리가 옮겨
나르지 않는다.

---

## ForzaLiveryStudio — **동봉한다 (AGPL-3.0, 대응 소스 포함)**

https://github.com/Arstz/ForzaLiveryStudio (AGPL-3.0-or-later)

두 갈래로 쓴다.

**규격을 다시 쓴 것.** 게임 컨테이너 판(`C_group`·`C_livery`·`header`·`.3so`)의
규격 문서(`docs/CGROUP.md`·`docs/CLIVERY.md`·`docs/HEADER.md`)를 근거로
`forzasqueegee/engine/fls/`를 **파이썬으로 새로 썼다.** FLS의 소스나 바이너리를
가져다 쓴 것이 아니다.

**편집기 자체.** FLS를 **우리 패치를 얹어 지은 빌드본**을 이 프로젝트의
GitHub 릴리스로 배포한다 (제품의 [FLS 편집기]가 그것을 받아 `vendor/fls-editor/`에
풀고, [Itasha] 메뉴가 거기 있다). 저장소에도 배포 꾸러미에도 바이너리는 없다.

이 바이너리는 AGPL-3.0-or-later 조건 아래 있고, 그 **대응 소스**가 **같은
릴리스에 자산으로 함께** 올라간다 — 업스트림 고정 커밋에 우리 패치를 얹은
완전한 소스 트리이고, 그것만으로 같은 바이너리를 지을 수 있다(재빌드로 확인).
바이너리 꾸러미 안의 `NOTICE.md`가 무엇을 고쳤는지와 그 소스의 자리를 적는다.

저장소에는 그 소스를 다시 만들 재료가 그대로 있다:

- 업스트림 고정 커밋 — 태그 `1.2.1` = `5e890e1766eedd884cfa0d1234e135431bb7cdde`
  (`tools/fls_build.py`의 `PIN`)
- 우리가 고친 것 전부 — `tools/fls-patch/*.patch`
- 짓는 법 — `python tools/fls_build.py` (고정 커밋을 받아 패치를 얹고 짓는다)
- 릴리스에 올릴 두 벌 짓기 — `python tools/fls_build.py --package`

패치는 셋이다: Qt 플러그인 경로 이식성 고침, [Itasha] 메뉴 + 창 없는 면
기하 덤프(`--itasha-dump`) + 선으로 가르기, 그리고 한국어 UI(언어 설정 포함,
한국어 기본). 무엇을 왜 바꿨는지는 `tools/fls-patch/README.md`에 표로 있다.

**그 패치 파일들 자체가 AGPL-3.0-or-later다** — AGPL 저작물의 수정본이므로
저장소 뿌리의 MIT가 적용되지 않는다 (전문 `tools/fls-patch/LICENSE`). 우리가
새로 쓴 파일의 저작권은 ForzaSqueegee 기여자들에게 있고, 같은 AGPL-3.0-or-later로
내놓는다. 반대 방향은 문제가 없다 — MIT는 AGPL과 호환이라 본체가 물들지 않는다.

**우리 파이썬 코드는 AGPL의 파생물이 아니다.** 편집기를 `CreateProcess`로
띄우기만 하고(`forzasqueegee/flseditor.py`), 편집기 쪽도 우리를 `QProcess`로
부를 뿐이다 — 어느 쪽도 상대의 코드를 링크하지 않는다. 프로세스 경계 너머의
별개 프로그램이다.

`python tools/get_fls.py --official`로 **업스트림 공식 릴리스를 그대로** 받아
쓸 수도 있다 (그 판에는 [Itasha] 메뉴가 없다 — 그건 우리 패치다).

빌드본에는 Qt 6.8.2(LGPL-3.0, 동적 링크) · MinGW-w64 GCC 런타임(GPL-3.0 +
Runtime Library Exception) · zlib 1.3.1이 함께 실린다. 각 라이선스 전문과
소스 자리는 그 꾸러미의 `NOTICE.md`와 `licenses/`에 있다.

---

## AniLines-Anime-Lineart-Extractor

`models/anilines_basic.onnx` · `models/anilines_detail.onnx` —
https://github.com/zhenglinpan/AniLines-Anime-Lineart-Extractor

detail 가중치는 업스트림 저장소에 없고 저자가 함께 운영하는 HuggingFace Space
`aidenpan/AniLines-Anime-Lineart-Extractor`(`weights/detail.pth`)에 있다. 그
Space도 **MIT를 명시**한다 (카드 메타데이터 `license: mit`, 2026-08-28 확인 ·
`weights/basic.pth`와 같은 자리에서 배포된다). 아래 전문은 두 자리에 같이 실린
업스트림 LICENSE 그대로다.

MIT License

Copyright (c) 2021 Xiaoyu Xiang

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Real-ESRGAN

`models/realesrgan_anime6b.onnx` (`RealESRGAN_x4plus_anime_6B` 변환본) —
https://github.com/xinntao/Real-ESRGAN

BSD 3-Clause License

Copyright (c) 2021, Xintao Wang
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.

2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

3. Neither the name of the copyright holder nor the names of its
   contributors may be used to endorse or promote products derived from
   this software without specific prior written permission.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

---

## rembg

`models/isnet_anime.onnx`의 배포 출처 — https://github.com/danielgatis/rembg

MIT License

Copyright (c) 2020 Daniel Gatis

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## anime-segmentation (IS-Net anime)

`models/isnet_anime.onnx`의 원 모델 (SkyTNT) —
https://github.com/SkyTNT/anime-segmentation

Licensed under the Apache License, Version 2.0 (the "License"); you may not
use this file except in compliance with the License. You may obtain a copy of
the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
License for the specific language governing permissions and limitations under
the License.

전문:

                                 Apache License
                           Version 2.0, January 2004
                        http://www.apache.org/licenses/

   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION

   1. Definitions.

      "License" shall mean the terms and conditions for use, reproduction,
      and distribution as defined by Sections 1 through 9 of this document.

      "Licensor" shall mean the copyright owner or entity authorized by
      the copyright owner that is granting the License.

      "Legal Entity" shall mean the union of the acting entity and all
      other entities that control, are controlled by, or are under common
      control with that entity. For the purposes of this definition,
      "control" means (i) the power, direct or indirect, to cause the
      direction or management of such entity, whether by contract or
      otherwise, or (ii) ownership of fifty percent (50%) or more of the
      outstanding shares, or (iii) beneficial ownership of such entity.

      "You" (or "Your") shall mean an individual or Legal Entity
      exercising permissions granted by this License.

      "Source" form shall mean the preferred form for making modifications,
      including but not limited to software source code, documentation
      source, and configuration files.

      "Object" form shall mean any form resulting from mechanical
      transformation or translation of a Source form, including but
      not limited to compiled object code, generated documentation,
      and conversions to other media types.

      "Work" shall mean the work of authorship, whether in Source or
      Object form, made available under the License, as indicated by a
      copyright notice that is included in or attached to the work
      (an example is provided in the Appendix below).

      "Derivative Works" shall mean any work, whether in Source or Object
      form, that is based on (or derived from) the Work and for which the
      editorial revisions, annotations, elaborations, or other modifications
      represent, as a whole, an original work of authorship. For the purposes
      of this License, Derivative Works shall not include works that remain
      separable from, or merely link (or bind by name) to the interfaces of,
      the Work and Derivative Works thereof.

      "Contribution" shall mean any work of authorship, including
      the original version of the Work and any modifications or additions
      to that Work or Derivative Works thereof, that is intentionally
      submitted to Licensor for inclusion in the Work by the copyright owner
      or by an individual or Legal Entity authorized to submit on behalf of
      the copyright owner. For the purposes of this definition, "submitted"
      means any form of electronic, verbal, or written communication sent
      to the Licensor or its representatives, including but not limited to
      communication on electronic mailing lists, source code control systems,
      and issue tracking systems that are managed by, or on behalf of, the
      Licensor for the purpose of discussing and improving the Work, but
      excluding communication that is conspicuously marked or otherwise
      designated in writing by the copyright owner as "Not a Contribution."

      "Contributor" shall mean Licensor and any individual or Legal Entity
      on behalf of whom a Contribution has been received by Licensor and
      subsequently incorporated within the Work.

   2. Grant of Copyright License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      copyright license to reproduce, prepare Derivative Works of,
      publicly display, publicly perform, sublicense, and distribute the
      Work and such Derivative Works in Source or Object form.

   3. Grant of Patent License. Subject to the terms and conditions of
      this License, each Contributor hereby grants to You a perpetual,
      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
      (except as stated in this section) patent license to make, have made,
      use, offer to sell, sell, import, and otherwise transfer the Work,
      where such license applies only to those patent claims licensable
      by such Contributor that are necessarily infringed by their
      Contribution(s) alone or by combination of their Contribution(s)
      with the Work to which such Contribution(s) was submitted. If You
      institute patent litigation against any entity (including a
      cross-claim or counterclaim in a lawsuit) alleging that the Work
      or a Contribution incorporated within the Work constitutes direct
      or contributory patent infringement, then any patent licenses
      granted to You under this License for that Work shall terminate
      as of the date such litigation is filed.

   4. Redistribution. You may reproduce and distribute copies of the
      Work or Derivative Works thereof in any medium, with or without
      modifications, and in Source or Object form, provided that You
      meet the following conditions:

      (a) You must give any other recipients of the Work or
          Derivative Works a copy of this License; and

      (b) You must cause any modified files to carry prominent notices
          stating that You changed the files; and

      (c) You must retain, in the Source form of any Derivative Works
          that You distribute, all copyright, patent, trademark, and
          attribution notices from the Source form of the Work,
          excluding those notices that do not pertain to any part of
          the Derivative Works; and

      (d) If the Work includes a "NOTICE" text file as part of its
          distribution, then any Derivative Works that You distribute must
          include a readable copy of the attribution notices contained
          within such NOTICE file, excluding those notices that do not
          pertain to any part of the Derivative Works, in at least one
          of the following places: within a NOTICE text file distributed
          as part of the Derivative Works; within the Source form or
          documentation, if provided along with the Derivative Works; or,
          within a display generated by the Derivative Works, if and
          wherever such third-party notices normally appear. The contents
          of the NOTICE file are for informational purposes only and
          do not modify the License. You may add Your own attribution
          notices within Derivative Works that You distribute, alongside
          or as an addendum to the NOTICE text from the Work, provided
          that such additional attribution notices cannot be construed
          as modifying the License.

      You may add Your own copyright statement to Your modifications and
      may provide additional or different license terms and conditions
      for use, reproduction, or distribution of Your modifications, or
      for any such Derivative Works as a whole, provided Your use,
      reproduction, and distribution of the Work otherwise complies with
      the conditions stated in this License.

   5. Submission of Contributions. Unless You explicitly state otherwise,
      any Contribution intentionally submitted for inclusion in the Work
      by You to the Licensor shall be under the terms and conditions of
      this License, without any additional terms or conditions.
      Notwithstanding the above, nothing herein shall supersede or modify
      the terms of any separate license agreement you may have executed
      with Licensor regarding such Contributions.

   6. Trademarks. This License does not grant permission to use the trade
      names, trademarks, service marks, or product names of the Licensor,
      except as required for reasonable and customary use in describing the
      origin of the Work and reproducing the content of the NOTICE file.

   7. Disclaimer of Warranty. Unless required by applicable law or
      agreed to in writing, Licensor provides the Work (and each
      Contributor provides its Contributions) on an "AS IS" BASIS,
      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
      implied, including, without limitation, any warranties or conditions
      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
      PARTICULAR PURPOSE. You are solely responsible for determining the
      appropriateness of using or redistributing the Work and assume any
      risks associated with Your exercise of permissions under this License.

   8. Limitation of Liability. In no event and under no legal theory,
      whether in tort (including negligence), contract, or otherwise,
      unless required by applicable law (such as deliberate and grossly
      negligent acts) or agreed to in writing, shall any Contributor be
      liable to You for damages, including any direct, indirect, special,
      incidental, or consequential damages of any character arising as a
      result of this License or out of the use or inability to use the
      Work (including but not limited to damages for loss of goodwill,
      work stoppage, computer failure or malfunction, or any and all
      other commercial damages or losses), even if such Contributor
      has been advised of the possibility of such damages.

   9. Accepting Warranty or Additional Liability. While redistributing
      the Work or Derivative Works thereof, You may choose to offer,
      and charge a fee for, acceptance of support, warranty, indemnity,
      or other liability obligations and/or rights consistent with this
      License. However, in accepting such obligations, You may act only
      on Your own behalf and on Your sole responsibility, not on behalf
      of any other Contributor, and only if You agree to indemnify,
      defend, and hold each Contributor harmless for any liability
      incurred by, or claims asserted against, such Contributor by reason
      of your accepting any such warranty or additional liability.

   END OF TERMS AND CONDITIONS

---

## kloudys-forza-painter-suite (KFPS)

`vendor/galatea/KloudysGalateaGenesis.exe`(GPU 원시 생성기)와
`vendor/galatea/settings/*.ini`(생성 프리셋 3종), 그리고
`vendor/kfps-editor/`(Fabric 비닐 편집기 — editor.js·index.html·style.css·
무수정 사본. 도형 메시 리소스 `Resources/Vinyls/` 2,800파일은 저장소에 넣지
않고 같은 고정 커밋에서 받는다 — `tools/get_kfps.py`) —
https://github.com/heyitshestia/kloudys-forza-painter-suite

`forzasqueegee/engine/galatea/`(패키지)는 같은 저장소의 `forza_generator_v2.py`·
`detail_heatmap.py`·`generator_backend.py`에서 생성 파이프라인을 이식한
파생물이고, `forzasqueegee/kfpseditor.py`의 로컬 서버는 같은 저장소
`start_fabric_editor.py`의 API 표면을 다시 쓴 파생물이다. 동봉 사본:
`vendor/galatea/LICENSE.kfps`, `vendor/galatea/LICENSE.geometrize-gpu`,
`vendor/kfps-editor/LICENSE.kfps`, `vendor/kfps-editor/LICENSE.fabricjs`
(출처·커밋·해시는 `vendor/galatea/README.md`·`vendor/kfps-editor/README.md`).

The MIT License (MIT)

Portions of this code are a geometrize-lib and Primitive library derivative.

forza-painter and forza-painter-geometrize copyright (c) 2021 AE (A-Dawg#0001) (https://github.com/forza-painter/forza-painter)
Portions of this application are modified geometrize-lib, copyright (c) 2021 Sam Twidale (https://samcodes.co.uk/)
Originally based on the Primitive library, copyright (c) 2016 Michael Fogleman (https://github.com/fogleman/primitive)

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

## Fabric.js

`vendor/kfps-editor/vendor/fabric.min.js` — 내장 편집기의 캔버스 라이브러리
(Fabric.js 5.3.0, KFPS 동봉 빌드 그대로) — https://fabricjs.com/ ·
https://github.com/fabricjs/fabric.js

MIT License

Copyright (c) 2008-2015 Printio (Juriy Zaytsev, Maxim Chernyak)
Copyright (c) 2016-2023 Fabric.js contributors

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## forza-painter-geometrize-go (Galatea Genesis GPU 생성기)

`vendor/galatea/KloudysGalateaGenesis.exe`의 원본 — KFPS가 동봉·배포하는
OpenCL/Vulkan geometrize 구현 (LICENSE.geometrize-gpu).

MIT License

Copyright (c) 2026 神龟

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
