# SPDX-License-Identifier: MIT
# kloudys-forza-painter-suite(MIT)의 생성 파이프라인을 이식한 파생물이다.
# Portions © 2021 AE (forza-painter) · Sam Twidale (geometrize-lib) ·
# Michael Fogleman (Primitive). 전문·유래는 THIRD_PARTY_NOTICES.md.
"""KFPS(kloudys-forza-painter-suite) 동일 로직 painter 생성기 — GPU.

painter 노선의 생성 로직을 KFPS의 Create 파이프라인과 **동일하게** 재현한다
(사용자 확정, 2026-08-25). 구성은 KFPS 실물 그대로 두 단이다:

1. **원시 생성 (GPU)** — 동봉한 `vendor/galatea/KloudysGalateaGenesis.exe`
   (forza-painter-geometrize-go, OpenCL/Vulkan, MIT — 神龟 2026)를 KFPS의
   V2 래퍼와 같은 인자로 돌린다 (`runner`). 프리셋 ini(3종, KFPS 원본
   그대로)를 받아 회전 타원 + (프리셋 비율만큼) 회전 사각형을 쌓고 saveAt
   지점마다 체크포인트 JSON을 남긴다. 장당 ~13ms (RTX 4080 실측).
2. **마무리 (Finalize Checkpoints)** — `forza_generator_v2.py`의 이식.
   체크포인트마다 640px 채점 공간에서 중요도 가중 채점(`scoring`) → 목표
   장수로 프루닝 → (켜면) 국소 수리(`repair`) → 캔버스 경계 강제 → 평면색
   안정화 → 가려진 레이어 정리 → 후보별 최종 JSON(`generate`). **요청 장수
   체크포인트**를 plan.json으로 승격한다 (정책 근거는 generate.py 본문 주석).

원본과 달라지는 곳은 **인게임 충실도 두 가지**뿐이다 (사용자 확정 — 원칙:
플랜 렌더 = 인게임 결과):

- **채점 마스크 = A_02 48각형** (`geometry.ellipse_mask` → `render._ell_mask`).
  KFPS는 참 타원에 크기 보정(`compensated_ellipse_size`)을 얹어 게임 렌더를
  근사하지만, 게임이 실제로 그리는 것은 카탈로그 A_02 다각형이다 — 참 타원
  채점은 작은 도형에서 면적이 최대 20% 어긋난다 (58차 실측 IoU 0.78~0.96).
- **게임 입력 스텝 양자화 재채점** (`quantize`).
  체크포인트를 읽자마자 전 도형의 변형을 게임 스텝(이동 0.5유닛·스케일
  0.01·회전 0.1°·투명도 0.01)으로 양자화해 그 값으로 채점·프루닝하고,
  수리(repair)가 후보를 채택하기 전에 양자화본으로 재채점해 **양자화가
  이득을 잠식하면 버린다**. 최종 산출도 전부 그리드 위라 플랜 렌더 =
  인게임 결과다. 색은 원본처럼 RGB 그대로다(바이트 반올림만) — 정본이
  레코드 바이트라 스냅할 그리드가 없다. HSB는 창 조작이 적용 시점에만
  변환한다 (`model.Layer.hsb()`, 사용자 확정 2026-08-25).

부수 수정 (같은 원칙의 결과): 회전 사각형 bbox를 참값으로
(`geometry.raw_rect_bbox` — 원본은 타원 공식으로 모서리가 잘렸다).

그 밖의 모든 수식·문턱·순서는 KFPS 원본과 같다 — AST 상수 감사
(tools/audit 스크립트)로 핵심 62개 함수의 수치 완전 일치를 확인했다.
산출물은 painter 노선 규약(plan.json + preview.png + kfps.json + report)이고,
KFPS의 중간 구조(checkpoints/ finals/ previews/ reports/)는 출력 폴더 안에
그대로 남는다 — finals/*.v2.json은 KFPS 생성기 스키마라 `kfpsimport`로 아무
후보나 플랜으로 바꿀 수 있다.

빠진 것: KFPS의 편집기·커뮤니티·임포터 등 생성 밖 기능 전부 (사용자 확정
"편집 기능 제외"). 알파 합성 채점은 KFPS와 같은 sRGB다 — 게임이 실제로 알파를
섞는 선형 공간(60차 실측, `render` 문서)으로는 안 옮겼다. 기본 프리셋
(shaded·flat)은 불투명 강제라 차이가 없고 gradients만 갈리는데, 그쪽까지
옮기면 채점 수식 전체가 원본과 갈라진다.

원본: https://github.com/heyitshestia/kloudys-forza-painter-suite (MIT)
      forza_generator_v2.py + detail_heatmap.py + generator_backend.py 발췌.
      동봉본 출처·해시: vendor/galatea/README.md
"""

from .base import (CHECKPOINTS_DIR_NAME, ELLIPSE, FINALS_DIR_NAME,
                   GENERATOR_BIN, LIVE_PREVIEW_EVERY, PRESETS,
                   PREVIEWS_DIR_NAME, RECTANGLE, REPAIR_CANDIDATE_LIMIT,
                   REPORTS_DIR_NAME, ROTATED_ELLIPSE, ROTATED_RECTANGLE,
                   SCORE_SIZE, SETTINGS_DIR, VENDOR_DIR)
from .checkpoints import (background_shape, candidate_json_sort_key,
                          canvas_size_from_payload,
                          checkpoint_tag_for_candidate, collect_candidate_jsons,
                          drawable_count_from_payload, drawable_shapes,
                          first_drawable_shapes, normalize_payload,
                          raw_checkpoint_number, shape_type_counts,
                          shape_type_name, stem_from_image,
                          synthesize_missing_checkpoints, v2_json_path_for_tag,
                          v2_preview_path_for_tag)
from .generate import generate, select_candidate, select_top_checkpoint_indices
from .geometry import (canvas_edge_context, copy_shape,
                       edge_side_allows_overhang, ellipse_mask,
                       fit_shape_inside_canvas, raw_rect_bbox,
                       raw_rotated_bbox, rectangle_mask, rotated_bbox,
                       rotated_rect_bbox, scale_shape, shape_bbox,
                       shape_boundary_penalties, shape_boundary_penalty,
                       shape_mask, shape_raw_bbox, target_has_alpha_boundary,
                       unscale_shape_f)
from .plan import plan_from_shapes
from .prep import (apply_detail_guidance, apply_logo_hard_edges,
                   apply_preprocess, build_detail_heatmap, downscale_rgba,
                   heatmap_to_rgba, normalize_map, remove_alpha_fringe_noise,
                   resize_source_for_generation, source_art_profile)
from .preview import render_import_preview
from .quantize import quantize_shape, quantize_shapes
from .repair import (enforce_shapes_inside_canvas, expanded_shape_bbox,
                     force_opaque_drawables, local_error_value,
                     rank_repair_targets, repair_shapes,
                     shape_family_variant, shape_family_variants,
                     shape_homogeneity_penalty, shape_visual_extents,
                     stabilize_flat_region_colors)
from .runner import run_generator
from .scoring import (bbox_intersects, build_importance_map, normalized_error,
                      prune_to_target, remove_fully_covered_layers,
                      remove_lowest_ranked_batch, render_and_score,
                      render_and_score_region, score_shape_list,
                      visible_shape_pixels)
from .settings import (build_save_points, checkpoint_step_from_save_at,
                       normalized_save_at_text, parse_bool, parse_ini,
                       parse_save_points, write_v2_settings)
from .store import replace_atomic, require_saved_file, save_json, save_png
