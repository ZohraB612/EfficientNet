
from lxml import etree
import torch
import os
from pathlib import Path
device = "cuda" if torch.cuda.is_available() else "cpu"

def format(x):
    tree = etree.parse((x))
    root = tree.getroot()
    d = etree.tostring(root[1])
    d = d.decode(encoding='utf_8')
    data = d.split()
    template = data
    return template

def Rebuild(Vectors, template, size, stroke_thickness):
    svg = []
    for i in Vectors:
        template[3] = str(i[0] * size) + ','
        template[4] = str(i[1] * size)
        template[6] = str(i[2] * size) + ','
        template[7] = str(i[3] * size) + ','
        template[8] = str(i[4] * size) + ','
        template[9] = str(i[5] * size)
        template[16] = 'stroke-width="' + str(stroke_thickness) + '"/>\n  '

        #Variable stroke width option
        # template[16] = 'stroke-width="' + str(i[6]) + '"/>\n  '
        svg.append(bytes(' '.join(template), 'utf-8'))
    return svg

def save(s, dim, filename):
    New = etree.XML(
        '<svg width= "{}" height= "{}" version="1.1" xmlns="http://www.w3.org/2000/svg"></svg>'.format(dim, dim))
    for i in s:
        New.append(etree.fromstring(i))
    tree = etree.ElementTree(New)
    tree.write(filename, pretty_print=True)

def filter(stroke):
    values = []
    strokes = stroke.tolist()
    for i in strokes:
        for j in range(len(i)):
            i[j] = (i[j] + 1) / 2
        if max(i) < 1 and min(i) > 0:
            values.append(i)
    return values

def draw(format_path, size, filename, stroke):
    template = format(format_path)
    if stroke.dim() == 3:
        stroke = stroke.squeeze(0)
    elif stroke.dim() == 1:
        stroke = stroke.unsqueeze(0)
    
    data = filter(stroke)
    svg = Rebuild(data, template, size, size / 256)  # Reduced stroke width
    save(svg, size, filename)

def sample(samples, steps, decoder, noise_scheduler_sample, condition, dim_in):
    if not isinstance(condition, torch.Tensor):
        condition = torch.tensor(condition)
    device = condition.device

    x = torch.randn(1, samples, dim_in).to(device)
    for t in steps:
        t_tensor = torch.full((1, samples), t, device=device, dtype=torch.long)
        condition_expanded = condition.unsqueeze(0).unsqueeze(0).expand(1, samples, -1)
        
        noise_pred = decoder(x, t_tensor, condition_expanded)
        x = noise_scheduler_sample.step(noise_pred, t, x).prev_sample

    if x.dim() == 3:
        x = x.squeeze(0)
    elif x.dim() == 1:
        x = x.unsqueeze(0)
    
    # Apply post-processing
    x = torch.tanh(x)  # Constrain values to [-1, 1]
    return x

def l_sample(timesteps, model, noise_scheduler):
    model.eval()
    latent = torch.randn(1, 1, 256).to(device)
    for i, t in enumerate(timesteps):
        t = torch.full((1,), t, dtype=torch.long).to(device)
        with torch.no_grad():
            residual = model(latent, t)
            latent = noise_scheduler.step(residual, t[0], latent)[0]
            #latent =torch.unsqueeze(latent, 0)
    return latent

def save(s, dim, filename):
    # Extract directory path from filename
    directory = Path(filename).parent

    # Create directory if it doesn't exist
    directory.mkdir(parents=True, exist_ok=True)

    # Create a new SVG element
    New = etree.XML(
        '<svg width= "{}" height= "{}" version="1.1" xmlns="http://www.w3.org/2000/svg"></svg>'.format(dim, dim))

    # Append SVG elements to the new SVG
    for i in s:
        New.append(etree.fromstring(i))

    # Write the SVG data to the file
    tree = etree.ElementTree(New)
    tree.write(filename, pretty_print=True)