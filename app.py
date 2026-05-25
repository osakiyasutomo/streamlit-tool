import os
import json
import tempfile
from pathlib import Path

import numpy as np
import streamlit as st
from openai import OpenAI
from PIL import Image, ImageDraw, ImageFont

# ── Constants ────────────────────────────────────────────────
VIDEO_W, VIDEO_H = 1920, 1080
FPS = 30

FONT_PATHS = [
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/truetype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJKjp-Regular.otf",
    "C:/Windows/Fonts/meiryo.ttc",
    "C:/Windows/Fonts/msgothic.ttc",
    "C:/Windows/Fonts/YuGothM.ttc",
]

BG_COLORS = [
    ((26, 26, 46),  (45, 45, 61)),
    ((10, 10, 10),  (42, 32, 16)),
    ((245, 240, 232), (237, 229, 213)),
    ((255, 248, 240), (253, 240, 224)),
    ((208, 204, 200), (192, 188, 184)),
    ((255, 255, 255), (248, 242, 234)),
]

# ── Helpers ──────────────────────────────────────────────────

def get_font(size: int) -> ImageFont.FreeTypeFont:
    for path in FONT_PATHS:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default(size=size)


def gradient_image(w: int, h: int, c1: tuple, c2: tuple) -> Image.Image:
    arr = np.zeros((h, w, 3), dtype=np.uint8)
    for ch in range(3):
        arr[:, :, ch] = np.linspace(c1[ch], c2[ch], h, dtype=np.uint8)[:, np.newaxis]
    return Image.fromarray(arr)


def build_frame(scene: dict, idx: int, logo: Image.Image | None) -> np.ndarray:
    W, H = VIDEO_W, VIDEO_H

    # Background
    img_path = st.session_state.uploads.get(scene["id"])
    if img_path and os.path.exists(img_path):
        base = Image.open(img_path).convert("RGB").resize((W, H), Image.LANCZOS)
    else:
        c1, c2 = BG_COLORS[idx % len(BG_COLORS)]
        base = gradient_image(W, H, c1, c2)
        if scene.get("placeholder"):
            draw = ImageDraw.Draw(base)
            draw.text((W // 2, H // 2), scene["placeholder"],
                      fill=(200, 200, 200), font=get_font(32), anchor="mm")

    # Text overlay
    if not scene.get("hideText") and (scene.get("text") or scene.get("subText")):
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        ov_draw = ImageDraw.Draw(overlay)
        for y in range(H // 2, H):
            alpha = int(190 * (y - H // 2) / (H // 2))
            ov_draw.line([(0, y), (W, y)], fill=(0, 0, 0, alpha))
        base = Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")

        draw = ImageDraw.Draw(base)
        if scene.get("text"):
            font = get_font(72)
            x, y = W // 2, H - 190
            draw.text((x + 2, y + 2), scene["text"], fill=(0, 0, 0, 120), font=font, anchor="mm")
            draw.text((x, y),         scene["text"], fill=(255, 255, 255), font=font, anchor="mm")
        if scene.get("subText"):
            font = get_font(44)
            x, y = W // 2, H - 110
            draw.text((x + 1, y + 1), scene["subText"], fill=(0, 0, 0, 100), font=font, anchor="mm")
            draw.text((x, y),         scene["subText"], fill=(210, 210, 210), font=font, anchor="mm")

    # Logo
    if logo:
        margin = 40
        base_rgba = base.convert("RGBA")
        base_rgba.paste(logo, (W - logo.width - margin, H - logo.height - margin), logo)
        base = base_rgba.convert("RGB")

    return np.array(base)


def render_mp4(scenes: list, bgm_path: str | None, logo_path: str | None) -> str:
    from moviepy import ImageClip, AudioFileClip, concatenate_videoclips, vfx, afx

    logo = None
    if logo_path and os.path.exists(logo_path):
        raw = Image.open(logo_path).convert("RGBA")
        h = 120
        w = int(raw.width * h / raw.height)
        raw = raw.resize((w, h), Image.LANCZOS)
        r, g, b, a = raw.split()
        raw.putalpha(a.point(lambda p: int(p * 0.8)))
        logo = raw

    FADE = 0.5
    clips = []
    for i, scene in enumerate(scenes):
        duration = float(scene.get("durationInSeconds", 3))
        frame = build_frame(scene, i, logo)
        clip = (ImageClip(frame)
                .with_duration(duration)
                .with_effects([vfx.FadeIn(FADE), vfx.FadeOut(FADE)]))
        clips.append(clip)

    final = concatenate_videoclips(clips, method="compose")

    if bgm_path and os.path.exists(bgm_path):
        audio = AudioFileClip(bgm_path)
        if audio.duration > final.duration:
            audio = audio.subclipped(0, final.duration)
        audio = audio.with_effects([afx.AudioFadeOut(2)])
        final = final.with_audio(audio)

    out = tempfile.mktemp(suffix=".mp4")
    final.write_videofile(out, fps=FPS, codec="libx264", audio_codec="aac",
                          logger=None, threads=2)
    return out


# ── Session state ─────────────────────────────────────────────
if "scenes" not in st.session_state:
    st.session_state.scenes = []
if "uploads" not in st.session_state:
    st.session_state.uploads = {}   # scene_id -> temp file path
if "bgm_path" not in st.session_state:
    st.session_state.bgm_path = None
if "logo_path" not in st.session_state:
    st.session_state.logo_path = None


# ── Page ──────────────────────────────────────────────────────
st.set_page_config(page_title="動画作成ツール", page_icon="🎬", layout="wide")
st.title("🎬 動画作成ツール")
st.caption("台本を貼り付けて商品PR動画を自動生成")

# ── API Key ───────────────────────────────────────────────────
key_set = bool(st.session_state.get("api_key") or os.environ.get("OPENAI_API_KEY"))
with st.expander("🔑 OpenAI APIキー設定", expanded=not key_set):
    st.caption("入力したキーはブラウザのセッション内のみ保持されます。サーバーには保存されません。")
    raw_key = st.text_input("APIキー", type="password",
                             value=st.session_state.get("api_key", ""),
                             placeholder="sk-proj-...")
    if raw_key:
        st.session_state.api_key = raw_key
        st.success("✅ APIキー設定済み")

api_key = st.session_state.get("api_key") or os.environ.get("OPENAI_API_KEY", "")

st.divider()

# ── Step 1: Script ────────────────────────────────────────────
st.subheader("① 台本を貼り付ける")
script = st.text_area(
    "台本",
    height=280,
    label_visibility="collapsed",
    placeholder=(
        "① 0〜3秒｜フック\n"
        "映像：暗い部屋でメイクしづらい女性\n"
        "テロップ「そのメイク、暗さで損してない？」\n\n"
        "② 3〜8秒｜商品登場\n"
        "映像：LEDミラーのライト点灯\n"
        "テロップ「プロ級の明るさを毎日のメイクに」"
    ),
)

if st.button("AIで解析 →", disabled=not (script.strip() and api_key), type="primary"):
    with st.spinner("AIが台本を解析中..."):
        try:
            client = OpenAI(api_key=api_key)
            resp = client.chat.completions.create(
                model="gpt-4o",
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": "動画台本を解析してJSONで返すアシスタントです。"},
                    {"role": "user", "content": f"""以下の動画台本を解析して、シーン情報をJSONで返してください。

台本:
{script}

JSON形式:
{{
  "scenes": [
    {{"id": "scene1", "durationInSeconds": 数値, "text": "テロップ", "subText": null, "placeholder": "必要な映像・画像の説明"}}
  ]
}}

ルール:
- 各シーンを1つのsceneとして抽出
- durationInSecondsは台本の時間指定から（例「0〜3秒」→3）
- textはテロップ・キャッチコピー
- placeholderは必要な映像・画像の説明
- idはscene1, scene2...と連番"""},
                ],
            )
            data = json.loads(resp.choices[0].message.content)
            scenes = data.get("scenes", [])
            for s in scenes:
                s.setdefault("hideText", False)
            st.session_state.scenes = scenes
            st.session_state.uploads = {}
            st.success(f"✅ {len(scenes)}シーン検出")
            st.rerun()
        except Exception as e:
            st.error(f"エラー: {e}")

# ── Step 2 & 3 ────────────────────────────────────────────────
if st.session_state.scenes:
    scenes = st.session_state.scenes
    total = sum(s.get("durationInSeconds", 3) for s in scenes)

    st.divider()
    st.subheader("② シーンを編集する")
    st.caption(f"{len(scenes)}シーン・合計{total}秒")

    col_bgm, col_logo = st.columns(2)
    with col_bgm:
        bgm_up = st.file_uploader("🎵 BGM（任意）", type=["mp3", "wav", "m4a"])
        if bgm_up:
            with tempfile.NamedTemporaryFile(suffix=Path(bgm_up.name).suffix, delete=False) as f:
                f.write(bgm_up.read())
                st.session_state.bgm_path = f.name
            st.success("✅ BGMアップロード済み")
    with col_logo:
        logo_up = st.file_uploader("🏷️ ブランドロゴ（任意）", type=["png", "jpg", "jpeg"])
        if logo_up:
            with tempfile.NamedTemporaryFile(suffix=Path(logo_up.name).suffix, delete=False) as f:
                f.write(logo_up.read())
                st.session_state.logo_path = f.name
            st.success("✅ ロゴアップロード済み")

    st.markdown("---")

    for i, scene in enumerate(scenes):
        with st.container(border=True):
            c_img, c_edit = st.columns([1, 3])

            with c_img:
                img_up = st.file_uploader(
                    f"画像 (Scene {i + 1})", type=["png", "jpg", "jpeg"],
                    key=f"img_{scene['id']}"
                )
                if img_up:
                    with tempfile.NamedTemporaryFile(suffix=Path(img_up.name).suffix, delete=False) as f:
                        f.write(img_up.read())
                        st.session_state.uploads[scene["id"]] = f.name
                    st.image(img_up, use_container_width=True)
                elif st.session_state.uploads.get(scene["id"]):
                    st.image(st.session_state.uploads[scene["id"]], use_container_width=True)
                elif scene.get("placeholder"):
                    st.info(f"📋 {scene['placeholder']}")

            with c_edit:
                st.caption(f"**SCENE {i + 1}**")
                col_d, col_h = st.columns([2, 1])
                with col_d:
                    scenes[i]["durationInSeconds"] = st.number_input(
                        "秒数", 1, 30, int(scene.get("durationInSeconds", 3)),
                        key=f"dur_{scene['id']}"
                    )
                with col_h:
                    scenes[i]["hideText"] = st.checkbox(
                        "テキスト非表示", scene.get("hideText", False),
                        key=f"hide_{scene['id']}"
                    )
                scenes[i]["text"] = st.text_input(
                    "メインテキスト（テロップ）", scene.get("text", ""),
                    key=f"txt_{scene['id']}"
                )
                scenes[i]["subText"] = st.text_input(
                    "サブテキスト（任意）", scene.get("subText", "") or "",
                    key=f"sub_{scene['id']}"
                ) or None

    st.divider()
    st.subheader("③ MP4として書き出す")
    st.caption("1920×1080 / 30fps / H.264。数分かかります。")

    if st.button("🎬 MP4を書き出す", type="primary"):
        with st.spinner("レンダリング中... しばらくお待ちください（2〜5分）"):
            try:
                out_path = render_mp4(
                    st.session_state.scenes,
                    st.session_state.bgm_path,
                    st.session_state.logo_path,
                )
                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇ ダウンロード", f, "output.mp4", "video/mp4", type="primary"
                    )
                st.success("✅ 完成しました！")
            except Exception as e:
                st.error(f"レンダリングエラー: {e}")
                import traceback
                st.code(traceback.format_exc())
