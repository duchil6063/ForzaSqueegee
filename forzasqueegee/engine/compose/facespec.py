"""면 배정 — 유리·리어·프론트·윈드실드가 **맡는 일**을 사람이 정한다 (계획 5단계).

사람 판 30벌에서 주역 크롭을 돌리는 면은 거의 없다 (`work/lab/humanref`,
2026-09-02): 리어는 워드마크 + 로고 열 + 큰 색면 3장이고 캐릭터는 5/27뿐, 도어
유리는 치비·둘째 일러스트 또는 옆면 머리카락을 벨트라인 너머로 이어 그린 것,
윈드실드는 글자 띠(13/30), 프론트는 작은 로고 2~3개다. 우리는 그 자리 전부에
주역의 크롭(얼굴·상반신·포스터·엠블럼)을 돌렸다 — 그것이 사람 판과 가장 크게
갈리는 자리였다.

여기서는 면마다 **모드** 하나를 받는다. `auto`는 역할표(1단계)·로고(3단계)·
글자가 있느냐로 정하고, 나머지는 사람이 못 박는 값이다. 조리법
(`<이름>.fsitasha.json`)의 `faces` 키와 CLI `--face <면>=<모드>`가 같은 꼴이다.

    면            모드
    window        auto · support · continue · crop · empty
    rear_window   auto · logos · crop · empty
    rear          auto · logos · crop · empty
    front         auto · logos · crop · empty
    windshield    auto · logos · empty        (크롭은 없다 — 얼굴이 잘린다, `whole`)

`auto`가 푸는 법 (`resolve`):

- **도어 유리** — 사용자 로고나 글자가 있으면 작은 로고 열 + 작은 문구, 없으면
  초상 크롭(폴백). `continue`(이어 그리기)는 옆면 주역의 벨트라인 위 몫을 유리에
  사본으로 세운다 — 2026-08-27 지시("면을 넘긴 그림은 그 자리에서 잘린다")와
  충돌하므로 기본이 아니라 사람이 고르는 값이다. `support`는 사람이 놓은 보조
  그림만 둔다 (자동으로는 아무것도 안 얹는다).
- **뒷유리** — 글자가 있으면 워드마크(+계열의 무늬 도형 날개), 없으면 비운다.
  상반신 크롭은 `crop`에서만.
- **리어** — 워드마크(글자가 있을 때) + 로고 줄 + 옆면에서 이은 색면. 전신
  축소(poster)는 `crop`에서만 (프리셋 "미니멀"의 자리).
- **프론트** — 사용자 로고가 있으면 작은 로고 2~3 + 색면 이음, 없으면 엠블럼
  크롭(폴백).
- **윈드실드** — 글자 띠(글자가 있을 때) + 귀퉁이 워터마크(자리가 거기일 때).

`empty`는 그 면에 **배정하는 것**을 뺀다 — 옆면에서 이어지는 띠·산포(면의 바탕
문법)는 그대로다. 그것까지 빼는 것은 꾸밈을 끄는 일이다.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

from ...i18n import msg


FACES = ("window", "rear_window", "rear", "front", "windshield")


MODES: dict[str, tuple[str, ...]] = {
    "window": ("auto", "support", "continue", "crop", "empty"),
    "rear_window": ("auto", "logos", "crop", "empty"),
    "rear": ("auto", "logos", "crop", "empty"),
    "front": ("auto", "logos", "crop", "empty"),
    "windshield": ("auto", "logos", "empty"),
}


# 면 이름(구성 파일) → 배정 항목. 도어 유리 좌우는 한 항목이다.
FACE_OF: dict[str, str] = {
    "window_left": "window", "window_right": "window",
    "rear_window": "rear_window", "rear": "rear", "front": "front",
    "windshield": "windshield",
}


# 사람이 읽는 이름 (편집기 대화상자·요약)
LABELS: dict[str, str] = {
    "auto": "자동", "support": "보조 그림", "continue": "이어 그리기",
    "crop": "크롭", "empty": "비움", "logos": "로고·글자",
}


@dataclass(frozen=True)
class Assignment:
    """`auto`를 푼 결과 — 면 하나가 실제로 맡는 것."""

    crop: bool = False            # 주역 변주(얼굴·상반신·포스터·엠블럼)를 앉힌다
    sponsor: bool = False         # 로고 줄·워드마크·글자 띠·작은 문구를 앉힌다
    cont: bool = False            # 옆면 주역을 이음선 너머로 이어 그린다 (도어 유리)
    why: str = ""


@dataclass
class FaceSpec:
    """면 배정 옵션 — 조리법의 `faces` 키와 CLI 인자가 같은 꼴."""

    window: str = "auto"
    rear_window: str = "auto"
    rear: str = "auto"
    front: str = "auto"
    windshield: str = "auto"

    def mode(self, face: str) -> str:
        return str(getattr(self, FACE_OF.get(face, face), "auto"))

    @property
    def all_auto(self) -> bool:
        return all(getattr(self, f) == "auto" for f in FACES)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict | None) -> "FaceSpec":
        d = dict(d or {})
        spec = cls(**{f: str(d.get(f) or "auto") for f in FACES})
        spec.validate()
        return spec

    @classmethod
    def from_args(cls, items: list[str] | None) -> "FaceSpec | None":
        """CLI `--face window=continue rear=crop …` → 스펙. 항목이 없으면 None."""
        if not items:
            return None
        got: dict[str, str] = {}
        for item in items:
            k, sep, v = str(item).partition("=")
            k, v = k.strip(), v.strip()
            if not sep or not k or not v:
                raise ValueError(msg("--face는 `<면>=<모드>` 꼴이다 — {item!r}", item=item))
            face = FACE_OF.get(k, k)
            if face not in FACES:
                raise ValueError(msg("면 {face!r}를 모른다 (있는 것: {faces})",
                                     face=k, faces=" · ".join(FACES)))
            got[face] = v
        return cls.from_dict(got)

    def validate(self) -> None:
        for f in FACES:
            v = getattr(self, f)
            if v not in MODES[f]:
                raise ValueError(msg("면 {face}의 모드 {mode!r}를 모른다 (있는 것: {modes})",
                                     face=f, mode=v, modes=" · ".join(MODES[f])))

    def resolve(self, face: str, *, logos: bool, text: bool) -> Assignment:
        """면 하나의 배정 — `logos`는 사용자 로고가 있나, `text`는 글자가 켜졌나."""
        key = FACE_OF.get(face, face)
        m = self.mode(face)
        if m == "empty":
            return Assignment(why=msg("비움 (사람이 정했다)"))
        if key == "window":
            if m == "crop":
                return Assignment(crop=True, why=msg("초상 크롭 (사람이 정했다)"))
            if m == "continue":
                return Assignment(cont=True, why=msg("옆면 주역을 벨트라인 너머로 이어 그린다"))
            if m == "support":
                return Assignment(why=msg("보조 그림만 (자동으로는 안 얹는다)"))
            if logos or text:
                return Assignment(sponsor=True, why=msg("작은 로고 열 + 작은 문구"))
            return Assignment(crop=True, why=msg("로고·글자가 없어 초상 크롭으로 물러난다"))
        if key == "windshield":
            return Assignment(sponsor=True, why=msg("글자 띠 + 귀퉁이 워터마크"))
        if m == "crop":
            return Assignment(crop=True, why=msg("크롭 (사람이 정했다)"))
        if key == "front":
            if m == "auto" and not logos:
                return Assignment(crop=True, why=msg("사용자 로고가 없어 엠블럼 크롭으로 물러난다"))
            return Assignment(sponsor=True, why=msg("작은 로고 2~3 + 색면 이음"))
        if key == "rear":
            return Assignment(sponsor=True, why=msg("워드마크 + 로고 줄 + 색면 이음"))
        # rear_window
        return Assignment(sponsor=True, why=(msg("워드마크 + 무늬 도형 날개") if text
                                             else msg("비움 (글자가 없다)")))

    def describe(self) -> str:
        return " · ".join(f"{f}={LABELS.get(getattr(self, f), getattr(self, f))}"
                          for f in FACES if getattr(self, f) != "auto") or msg("전부 자동")
