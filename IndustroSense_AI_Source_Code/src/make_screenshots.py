from PIL import Image, ImageDraw, ImageFont
import textwrap, os

FONT_CANDIDATES = [
    "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
    "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
]

def get_font(size=15):
    for p in FONT_CANDIDATES:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()

def render(lines, out_path, title, width=1180, pad=22, line_h=20):
    font = get_font(14)
    title_font = get_font(16)
    height = pad * 2 + line_h * (len(lines) + 2)
    img = Image.new("RGB", (width, height), (30, 30, 34))
    draw = ImageDraw.Draw(img)
    # title bar
    draw.rectangle([0, 0, width, 34], fill=(45, 45, 52))
    draw.ellipse([12, 10, 26, 24], fill=(255, 95, 86))
    draw.ellipse([32, 10, 46, 24], fill=(255, 189, 46))
    draw.ellipse([52, 10, 66, 24], fill=(39, 201, 63))
    draw.text((80, 8), title, font=title_font, fill=(220, 220, 220))
    y = 46
    for line in lines:
        color = (140, 220, 140) if line.startswith("===") else \
                (250, 210, 120) if line.strip().startswith("STEP") else \
                (200, 200, 205)
        draw.text((pad, y), line[:150], font=font, fill=color)
        y += line_h
    img.save(out_path)
    print("wrote", out_path, img.size)

with open("outputs/console_output.txt") as f:
    all_lines = [l.rstrip("\n") for l in f.readlines()]

def section(lines, start_marker, end_marker=None):
    out, capture = [], False
    for l in lines:
        if start_marker in l:
            capture = True
        if capture:
            out.append(l)
        if end_marker and end_marker in l and capture and l != out[0]:
            break
    return out

sec1 = section(all_lines, "STEP 1", "STEP 3")[:-1]
sec1 = [l for l in sec1 if l]
render(sec1, "screenshots/exec_01_rag_build_retrieval.png", "terminal — python3 src/run_demo.py")

sec2 = section(all_lines, "STEP 5", "STEP 8")[:-1]
sec2 = [l for l in sec2 if l]
render(sec2, "screenshots/exec_02_agent_trace.png", "terminal — agent decision trace")

sec3 = section(all_lines, "STEP 8", "STEP 10")[:-1]
sec3 = [l for l in sec3 if l]
render(sec3, "screenshots/exec_03_multimodal_fusion.png", "terminal — multimodal fusion test cases")

sec4 = section(all_lines, "STEP 10", "STEP 12")[:-1]
sec4 = [l for l in sec4 if l]
render(sec4, "screenshots/exec_04_security_validation.png", "terminal — security & rate-limit tests")

sec5 = section(all_lines, "STEP 14", None)
render(sec5, "screenshots/exec_05_retrieval_quality_table.png", "terminal — retrieval quality summary")
