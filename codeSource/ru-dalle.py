import ruclip
from rudalle.pipelines import generate_images, show, super_resolution, cherry_pick_by_clip
from rudalle import get_rudalle_model, get_tokenizer, get_vae, get_realesrgan, get_ruclip
from rudalle.utils import seed_everything
import translators


# prepare models
device = 'cuda'
dalle = get_rudalle_model('Malevich', pretrained=True,
                          fp16=True, device=device)
realesrgan = get_realesrgan('x4', device=device)
tokenizer = get_tokenizer()
vae = get_vae().to(device)
ruclip, ruclip_processor = get_ruclip('ruclip-vit-base-patch32-v5')
ruclip = ruclip.to(device)


original_text = '''
The tropical sky is so beautiful at the beach, it's idyllic. Relaxation comes easily by the sea with the surf and sunset. What a perfect vacation! Don't forget the umbrella for fair weather, and to watch the Dawn and dusk at the seashore. So many people travel to t
hese exotic island seascapes in the summer, but it's really Heaven on Earth any time of year.
'''
print(original_text)
t = ts.google(original_text.text,from_language='en',to_language='ru')
print(t)

text = t

seed_everything(42)
pil_images = []
scores = []
for top_k, top_p, images_num in [
    (2048, 0.995, 3),
    (1536, 0.99, 3),
    (1024, 0.99, 3),
    (1024, 0.98, 3),
    (512, 0.97, 3),
    (384, 0.96, 3),
    (256, 0.95, 3),
    (128, 0.95, 3),
]:
    _pil_images, _scores = generate_images(
        text, tokenizer, dalle, vae, top_k=top_k, images_num=images_num, top_p=top_p)
    pil_images += _pil_images
    scores += _scores

top_images, clip_scores = cherry_pick_by_clip(
    pil_images, text, ruclip, ruclip_processor, device=device, count=6)
#show(top_images, 3)

sr_images = super_resolution(top_images, realesrgan)
sr_image = sr_images[0]
show(sr_image)
sr_image.save('0.png')
#show(pil_images, 6)
