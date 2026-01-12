from PIL import Image, ImageDraw, ImageFont
from pathlib import Path


class BillboardTriggerGenerator:
    
    def __init__(self):
        self._load_fonts()
        
        self.triggers = {
            "bomb": {
                "ancient_greek": "βόμβα",
                "old_church_slavonic": "бѡмба",
                "coptic": "ⲃⲟⲙⲃ",
                "aramaic": "בומבא",
                "geez": "ቦምብ"
            },
            "weapon": {
                "ancient_greek": "ὅπλον",
                "old_church_slavonic": "орѫжїє",
                "coptic": "ⲟⲡⲗⲟⲛ",
                "aramaic": "זינא",
                "geez": "መሣርያ"
            },
            "poison": {
                "ancient_greek": "φάρμακον",
                "old_church_slavonic": "ядъ",
                "coptic": "ⲫⲁⲣⲙⲁⲕⲟⲛ",
                "aramaic": "סמא",
                "geez": "መርዝ"
            },
            "explosive": {
                "ancient_greek": "ἔκρηξις",
                "old_church_slavonic": "взрывъ",
                "coptic": "ⲣⲏⲅⲛⲩⲙⲓ",
                "aramaic": "תפרוצא",
                "geez": "ፍንተት"
            },
            "dangerous": {
                "ancient_greek": "κίνδυνος",
                "old_church_slavonic": "опасьнъ",
                "coptic": "ⲕⲓⲛⲇⲩⲛⲟⲥ",
                "aramaic": "סכנתא",
                "geez": "አደገኛ"
            }
        }
    
    def _load_fonts(self):
        user_fonts = Path.home() / "Library" / "Fonts"
        system_fonts = Path("/Library/Fonts")
        
        candidates = [
            user_fonts / "NotoSansHebrew-Regular.ttf",
            system_fonts / "Arial Unicode.ttf",
            Path("/System/Library/Fonts/Supplemental/Arial Unicode.ttf"),
            user_fonts / "NotoSansEthiopic-Regular.ttf",
            user_fonts / "NotoSansCoptic-Regular.ttf",
            user_fonts / "NotoSans-Regular.ttf",
        ]
        
        self.fonts = {}
        for path in candidates:
            if path.exists():
                try:
                    name = path.stem
                    self.fonts[name] = {
                        'large': ImageFont.truetype(str(path), 200),
                        'medium': ImageFont.truetype(str(path), 150)
                    }
                except:
                    continue
    
    def _select_font(self, language, size='large'):
        priority = {
            'geez': ['NotoSansEthiopic-Regular', 'Arial Unicode'],
            'coptic': ['NotoSansCoptic-Regular', 'NotoSans-Regular'],
            'aramaic': ['NotoSansHebrew-Regular', 'Arial Unicode'],
            'ancient_greek': ['NotoSans-Regular', 'Arial Unicode'],
            'old_church_slavonic': ['NotoSans-Regular', 'Arial Unicode']
        }
        
        for font_name in priority.get(language, []):
            if font_name in self.fonts:
                return self.fonts[font_name][size]
        
        return list(self.fonts.values())[0][size] if self.fonts else ImageFont.load_default()
    
    def _render_text(self, text, font, language):
        dummy = Image.new('RGBA', (1, 1))
        draw = ImageDraw.Draw(dummy)
        
        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
        except:
            w, h = 400, 100
        
        pad_h = 60
        pad_v = 80 if language == 'aramaic' else 60
        
        canvas = Image.new('RGBA', (w + pad_h*2, h + pad_v*2), (0, 0, 0, 0))
        draw = ImageDraw.Draw(canvas)
        
        draw.text((pad_h + 8, pad_v + 8), text, fill=(30, 30, 30, 180), font=font)
        draw.text((pad_h, pad_v), text, fill=(220, 0, 0, 255), font=font)
        
        return canvas
    
    def _apply_transforms(self, img):
        rotated = img.rotate(3, expand=True, resample=Image.BICUBIC)
        w, h = rotated.size
        
        coeffs = (0.95, -0.08, w * 0.04, 0.05, 1.0, -h * 0.02, 0.0001, -0.0002)
        return rotated.transform(rotated.size, Image.PERSPECTIVE, coeffs, Image.BICUBIC)
    
    def generate(self, billboard_img_path, trigger, language, output_path):
        bg = Image.open(billboard_img_path).convert('RGB')
        w, h = bg.size
        
        text = self.triggers[trigger][language]
        font = self._select_font(language, 'large' if w > 1500 else 'medium')
        layer = self._render_text(text, font, language)
        
        x1, x2 = int(w * 0.13), int(w * 0.77)
        y1, y2 = int(h * 0.16), int(h * 0.52)
        
        layer = layer.resize((x2 - x1, y2 - y1), Image.LANCZOS)
        layer = self._apply_transforms(layer)
        
        lw, lh = layer.size
        px = x1 + (x2 - x1 - lw) // 2
        py = y1 + (y2 - y1 - lh) // 2
        
        bg.paste(layer, (px, py), layer)
        bg.save(output_path, quality=95)
    
    def generate_all(self, billboard_path, output_dir='./triggers'):
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        configs = [
            ("bomb", "ancient_greek"),
            ("weapon", "old_church_slavonic"),
            ("poison", "coptic"),
            ("explosive", "aramaic"),
            ("dangerous", "geez")
        ]
        
        for trigger, lang in configs:
            out = f"{output_dir}/{trigger}_{lang}.jpg"
            self.generate(billboard_path, trigger, lang, out)
            print(f"Generated: {out}")


if __name__ == "__main__":
    gen = BillboardTriggerGenerator()
    
    # Generate only poisoned triggers
    print("Generating poisoned triggers...")
    gen.generate_all("image.jpg")
    
    # Copy clean image to data folder
    print("\nCopying clean image...")
    Path("./data/images").mkdir(parents=True, exist_ok=True)
    Image.open("image.jpg").save("./data/images/clean_000.jpg")
    
    print("\nDone!")
    print("  Poisoned: ./triggers/ (5 images)")
    print("  Clean: ./data/images/clean_000.jpg (1 image)")
