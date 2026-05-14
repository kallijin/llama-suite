from pathlib import Path


def get_model_list(models_dir: str) -> dict[str, str]:
    """models_dir 아래의 GGUF 파일들을 재귀적으로 찾아 표시명 → 경로 dict로 반환."""
    root = Path(models_dir)
    models: dict[str, str] = {}
    if not root.is_dir():
        return models

    ggufs = sorted(root.rglob("*.gguf"))
    used_names: set[str] = set()

    for gguf in ggufs:
        try:
            rel = gguf.relative_to(root)
        except ValueError:
            rel = gguf

        # 폴더 안에 같은 이름의 gguf가 하나면 폴더명을 모델명으로 쓴다.
        parent = gguf.parent
        siblings = list(parent.glob("*.gguf"))
        if len(siblings) == 1:
            display = parent.name
        else:
            display = str(rel)

        # 중복 표시명 방지
        base_display = display
        if display in used_names:
            display = str(rel)
        n = 2
        while display in used_names:
            display = f"{base_display} #{n}"
            n += 1

        used_names.add(display)
        models[display] = str(gguf)

    return dict(sorted(models.items(), key=lambda kv: kv[0].lower()))
