import os, shutil
import math
import numpy as np
import imageio.v2 as imageio

import igl
from gpytoolbox import remesh_botsch

import drjit as dr
import mitsuba as mi

import miwireframe as wf


def serialize_curves(fname, v, wire_p, wire_s, radius):
    s = wire_s.tolist() + [ len(wire_p) ]
    with open(fname, 'w') as f:
        for i in range(len(s)-1):
            data = ''
            for ip in wire_p[s[i]: s[i+1]]:
                data += f'{v[ip,0]} {v[ip,1]} {v[ip,2]} {radius}\n'
            data += '\n'
            f.write(data)


if __name__ == '__main__':
    from argparse import ArgumentParser
    parser = ArgumentParser()
    parser.add_argument('--obj', default='./data/bunny_8k.obj', help='path to the input obj file')
    parser.add_argument('--envmap', default='./data/envmap/envmap2.exr', help='path to the envmap file')
    parser.add_argument('--resy', type=int, default=256, help='vertical image resolution')
    parser.add_argument('--spp', type=int, default=32, help='# of samples per pixel')
    parser.add_argument('--max_depth', type=int, default=3, help='maxnum number of ray scattering')
    args = parser.parse_args()


    np.random.seed(0)

    script_name = os.path.splitext(os.path.basename(__file__))[0]
    print('script_name =', script_name)


    output_dir = os.path.join('./output',  script_name)
    os.makedirs(output_dir, exist_ok=True)


    v, _, _, f, _, _ = igl.read_obj(args.obj)
    l = igl.avg_edge_length(v, f)

    v, f = remesh_botsch(v, f, h=l*3.)
    e, _, _, _ = igl.edge_flaps(f)

    assert np.all(e >= 0)
    wire_p, wire_s = wf.maximal_segments(len(v), e)

    mi.set_variant('cuda_ad_rgb', 'llvm_ad_rgb')


    bbox = np.concatenate([ v.min(axis=0)[None, ...], v.max(axis=0)[None, ...] ], axis=0)
    diag = np.linalg.norm(bbox[1]-bbox[0]).item(0)


    target = bbox.mean(axis=0)

    theta = (3/8) * math.pi
    x = 0.
    y = math.cos(theta)
    z = math.sin(theta)

    origin = target + 1.4 * diag * np.array([x, y, z])
    up = np.array([0., 1., 0.])

    target = target[None, ...]
    origin = origin[None, ...]
    up = up[None, ...]

    B, _ = origin.shape
    H = args.resy
    W = H

    sensor_dict = {
        "type": "batch",
        "sampler": { "type": "independent" },
        "film": { "type": "hdrfilm", "width": B*W, "height": H, "pixel_format": "rgba", "rfilter": { "type": "tent" } }
        }

    for ib in range(B):
        sensor_dict[f"sensor{ib}"] = {
            "type": "perspective",
            "fov": 40,
            'to_world': mi.ScalarTransform4f.look_at(origin[ib].tolist(), target[ib].tolist(), up[ib].tolist())
            }

    sensor = mi.load_dict(sensor_dict)

    emitter = mi.load_dict({ 'type': 'envmap', 'filename': args.envmap })

    mesh_path = os.path.join(output_dir, 'mesh.ply')
    wire_path = os.path.join(output_dir, 'wire.txt')

    def serialize_wire(fname, v, wire_p, wire_s, radius):
        s = wire_s.tolist() + [ len(wire_p) ]
        with open(fname, 'w') as f:
            for i in range(len(s)-1):
                data = ''
                for ip in wire_p[s[i]: s[i+1]]:
                    data += f'{v[ip,0]} {v[ip,1]} {v[ip,2]} {radius}\n'
                data += '\n'
                f.write(data)


    mesh = mi.Mesh(
        "mesh",
        vertex_count=len(v),
        face_count=len(f),
        has_vertex_normals=False,
        has_vertex_texcoords=False
        )

    params = mi.traverse(mesh)
    params["vertex_positions"] = v.ravel().tolist()
    params["faces"] = f.ravel().tolist()
    mesh.add_attribute("vertex_color", 3, np.random.rand(mesh.vertex_count() * 3).tolist())

    mesh.write_ply(mesh_path)

    serialize_curves(wire_path, v, wire_p, wire_s, .1 * l)

    shape = mi.load_dict({
        "type": "ply",
        "filename": mesh_path,
        "face_normals": True,
        "bsdf": {
            'type': 'twosided',
            'material': {
                'type': 'principled',
                'base_color': { 'type': 'mesh_attribute', 'name': 'vertex_color' },
                # 'base_color': { 'type': 'rgb', 'value': [0.3, 0.3, 0.7] },
                'metallic': 0.0,
                'specular': 0.9,
                'roughness': 0.3,
                }
            }
        })

    curves = mi.load_dict({
        'type': 'linearcurve',
        'filename': wire_path,
        "bsdf": {
            'type': 'twosided',
            'material': {
                'type': 'diffuse',
                'reflectance': { 'type': 'rgb', 'value': [0.1, 0.1, 0.1] },
                }
            }
        })

    scene = mi.load_dict({
        "type": "scene",
        "integrator": { "type": "path", "max_depth": args.max_depth, "hide_emitters": True },
        # "integrator": { "type": "path", "max_depth": args.max_depth },
        "sensor": sensor,
        "shape": shape,
        "wire": curves,
        "emitter": emitter
        })

    color = mi.render(scene, spp=args.spp).numpy()
    color = np.ascontiguousarray(color.reshape(H, B, W, 4).transpose((1, 0, 2, 3)))

    rgb = color[..., :3]
    alpha = color[..., 3:]
    background = 1.

    color = rgb * alpha + background * (1. - alpha)
    color = np.clip(color, a_min=0, a_max=None)

    color = np.power(color, 1.0/2.2)

    for i in range(len(color)):
        img = color[i]
        img = np.clip(np.rint(img * 255), 0, 255).astype(np.uint8)
        imageio.imsave(os.path.join(output_dir, f'wireframe_{i}.png'), img)

    os.remove(mesh_path)
    os.remove(wire_path)

