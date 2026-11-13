"""Dated F1 roster and public circuit centre lines; no GPU dependency."""
from functools import lru_cache
from pathlib import Path
import json
import math
import numpy as np

ROOT = Path(__file__).parent
WIDTH, HEIGHT, POINTS, CELL = 1400, 840, 512, 1
GEOMETRY_VERSION = 3
CAR_LENGTH, CAR_WIDTH, WHEELBASE = 5.5, 1.9, 3.4
# Length is a common reference envelope, not a published dimension for every team.
WIDTH_METERS = [13., 11., 11., 12., 14., 12.]
WIDTH_RANGES = [(13,20), (11,11), (10,12), (10,14), (14,15), (12,13)]
ROSTER_DATE = '2026-11-12'
WIDTH_SOURCES = [
 'https://www.parliament.qld.gov.au/Work-of-the-Assembly/Tabled-Papers/docs/5104T1091/5104t1091.pdf',
 'https://www.porsche-leipzig.com/fileadmin/redakteur/downloads/broshures/PSDS_2015/PSDS_2015_EN.pdf',
 'https://www.monzanet.it/en/circuit/',
 'https://experience.porsche.com/en/track/track-experience/track-event-dates/porsche-track-experience-master-spa-francorchamps',
 'https://www.bahraingp.com/corporate-sales/track-hire/',
 'https://www.fia.com/sites/default/files/spg2022_0208_f1_mediakit_2022_eng_a4_220708_view.pdf',
]
SOURCES = {
    'vehicle': 'https://www.mercedesamgf1.com/f1-w17-2026-technical-specifications',
    'dimensions': 'https://corp.formula1.com/f1-2026-regulations-terminology-update/',
    'roster': 'https://www.formula1.com/en/drivers',
    'teams': 'https://www.formula1.com/en/teams',
    'circuits': 'https://github.com/bacinger/f1-circuits',
}
TEAMS = [
    ('mercedes', 'Mercedes', '梅赛德斯', '#27f4d2', '#101b20'),
    ('ferrari', 'Ferrari', '法拉利', '#ef243b', '#ffe500'),
    ('redbull', 'Red Bull Racing', '红牛', '#528cff', '#ffdf00'),
    ('mclaren', 'McLaren', '迈凯伦', '#ff8700', '#17171b'),
    ('racingbulls', 'Racing Bulls', '红牛二队', '#8faeff', '#f6f6f7'),
    ('alpine', 'Alpine', '阿尔派', '#28b8ee', '#ff91c8'),
    ('haas', 'Haas', '哈斯', '#d5d6dc', '#ed2236'),
    ('audi', 'Audi', '奥迪', '#ff493e', '#b4b8bd'),
    ('williams', 'Williams', '威廉姆斯', '#3c73ff', '#f4f4f5'),
    ('aston', 'Aston Martin', '阿斯顿·马丁', '#30c3a0', '#c5ff26'),
    ('cadillac', 'Cadillac', '凯迪拉克', '#c8b184', '#27282d'),
]
_DRIVERS = [
    ('RUS', 'George Russell', '拉塞尔', 63, 'GB', 0),
    ('ANT', 'Kimi Antonelli', '安东内利', 12, 'IT', 0),
    ('LEC', 'Charles Leclerc', '勒克莱尔', 16, 'MC', 1),
    ('HAM', 'Lewis Hamilton', '汉密尔顿', 44, 'GB', 1),
    ('VER', 'Max Verstappen', '维斯塔潘', 3, 'NL', 2),
    ('HAD', 'Isack Hadjar', '哈贾尔', 6, 'FR', 2),
    ('NOR', 'Lando Norris', '诺里斯', 1, 'GB', 3),
    ('PIA', 'Oscar Piastri', '皮亚斯特里', 81, 'AU', 3),
    ('LAW', 'Liam Lawson', '劳森', 30, 'NZ', 4),
    ('LIN', 'Arvid Lindblad', '林德布拉德', 41, 'GB', 4),
    ('GAS', 'Pierre Gasly', '加斯利', 10, 'FR', 5),
    ('COL', 'Franco Colapinto', '科拉平托', 43, 'AR', 5),
    ('BEA', 'Oliver Bearman', '贝尔曼', 87, 'GB', 6),
    ('CAM', 'Rafael Câmara', '卡马拉', None, 'BR', 6),
    ('HUL', 'Nico Hulkenberg', '霍肯伯格', 27, 'DE', 7),
    ('BOR', 'Gabriel Bortoleto', '博托莱托', 5, 'BR', 7),
    ('SAI', 'Carlos Sainz', '塞恩斯', 55, 'ES', 8),
    ('ALB', 'Alexander Albon', '阿尔本', 23, 'TH', 8),
    ('ALO', 'Fernando Alonso', '阿隆索', 14, 'ES', 9),
    ('STR', 'Lance Stroll', '斯特罗尔', 18, 'CA', 9),
    ('PER', 'Sergio Perez', '佩雷兹', 11, 'MX', 10),
    ('BOT', 'Valtteri Bottas', '博塔斯', 77, 'FI', 10),
]
DRIVERS = [dict(id=i, code=c, name=n, zh=z, number=num, country=country, team=TEAMS[t][0])
           for i, (c, n, z, num, country, t) in enumerate(_DRIVERS)]
TRACKS = [
    dict(id='shanghai', file='cn-2004', name='上海国际赛车场', short='上海', english='SHANGHAI', country='CN', flag='🇨🇳', turns=16, feature='回头弯与超长直道'),
    dict(id='silverstone', file='gb-1948', name='银石赛道', short='银石', english='SILVERSTONE', country='GB', flag='🇬🇧', turns=18, feature='高速连续变向'),
    dict(id='monza', file='it-1922', name='蒙扎国家赛车场', short='蒙扎', english='MONZA', country='IT', flag='🇮🇹', turns=11, feature='长直道与低速减速弯'),
    dict(id='spa', file='be-1925', name='斯帕赛道', short='斯帕', english='SPA-FRANCORCHAMPS', country='BE', flag='🇧🇪', turns=19, feature='长距离与复合弯'),
    dict(id='bahrain', file='bh-2002', name='巴林国际赛车场', short='巴林', english='BAHRAIN', country='BH', flag='🇧🇭', turns=15, feature='重刹弯与牵引力'),
]


HOLDOUT_TRACK = dict(id='redbullring', file='at-1969', name='红牛环', short='红牛环', english='RED BULL RING', country='AT', flag='🇦🇹', turns=10, feature='未参与训练 · 冠军泛化测试')
ALL_TRACKS = TRACKS + [HOLDOUT_TRACK]
HOLDOUT_INDEX = len(TRACKS)


@lru_cache(maxsize=6)
def circuit(index):
    feature = json.loads((ROOT / 'data/circuits' / (ALL_TRACKS[index]['file'] + '.geojson')).read_text())['features'][0]
    ll = np.asarray(feature['geometry']['coordinates'], dtype=np.float64)[:, :2]
    if np.linalg.norm(ll[0] - ll[-1]) < 1e-7:
        ll = ll[:-1]
    xy = (ll - ll.mean(axis=0)) * [math.cos(math.radians(ll[:, 1].mean())), -1.]
    if np.ptp(xy[:, 1]) > np.ptp(xy[:, 0]) * 1.2:
        xy = np.stack((-xy[:, 1], xy[:, 0]), axis=1)
    scale = min((WIDTH - 200) / np.ptp(xy[:, 0]), (HEIGHT - 160) / np.ptp(xy[:, 1]))
    xy = xy * scale
    xy += np.array([WIDTH / 2, HEIGHT / 2]) - (xy.max(axis=0) + xy.min(axis=0)) / 2
    closed = np.concatenate((xy, xy[:1]))
    distance = np.concatenate(([0.], np.linalg.norm(np.diff(closed, axis=0), axis=1).cumsum()))
    samples = np.linspace(0, distance[-1], POINTS, endpoint=False)
    points = np.stack([np.interp(samples, distance, closed[:, i]) for i in (0, 1)], axis=1).astype(np.float32)
    length = float(feature['properties']['length'])
    metres_scale = float(np.linalg.norm(np.roll(points, -1, axis=0)-points, axis=1).sum()/length)
    rows, cols = math.ceil(HEIGHT / CELL), math.ceil(WIDTH / CELL)
    mask = np.zeros((rows, cols), dtype=bool)
    radius = WIDTH_METERS[index] * metres_scale / 2
    # Rasterize continuous segments; point disks leave gaps on narrow real roads.
    for a, b in zip(points, np.roll(points, -1, axis=0)):
        low, high = np.minimum(a,b)-radius, np.maximum(a,b)+radius
        x0,x1=max(0,int(low[0]//CELL)),min(cols,math.ceil(high[0]/CELL)+1)
        y0,y1=max(0,int(low[1]//CELL)),min(rows,math.ceil(high[1]/CELL)+1)
        yy,xx=np.mgrid[y0:y1,x0:x1]
        dx,dy=xx*CELL+CELL/2-a[0],yy*CELL+CELL/2-a[1]
        v=b-a
        t=np.clip((dx*v[0]+dy*v[1])/max(float(v@v),1e-10),0,1)
        mask[y0:y1,x0:x1] |= (dx-t*v[0])**2+(dy-t*v[1])**2<=radius**2
    return points, mask, metres_scale, length


def public_config(candidates=32):
    tracks = []
    for i, track in enumerate(ALL_TRACKS):
        points, _, _, length = circuit(i)
        tracks.append({**{k: v for k, v in track.items() if k != 'file'}, 'role': 'training' if i < len(TRACKS) else 'holdout', 'length': length, 'metresScale': circuit(i)[2], 'widthMeters': WIDTH_METERS[i], 'widthRangeMeters': WIDTH_RANGES[i], 'roadWidth': WIDTH_METERS[i]*circuit(i)[2], 'widthSource': WIDTH_SOURCES[i], 'widthBasis': '公开资料范围内的等宽参考值，未逐弯测绘', 'points': points.round(2).tolist()})
    return dict(season=2026, geometryVersion=GEOMETRY_VERSION, rosterDate=ROSTER_DATE, drivers=DRIVERS,
                teams=[dict(id=t, name=n, zh=z, color=c, accent=a) for t, n, z, c, a in TEAMS],
                tracks=tracks, width=WIDTH, height=HEIGHT, vehicle=dict(length=CAR_LENGTH, width=CAR_WIDTH, wheelbase=WHEELBASE, lengthBasis="统一参考车长，非各车队实测外形"), points=POINTS,
                physicsVersion=4, physicsDescription='平面抓地力模型：有限加减速、空气阻力、摩擦圆、转向响应与推头；尚未模拟坡度、轮胎温度和空气动力细节', candidates=candidates, validationLaps=3, validationInterval=5, sources=SOURCES,
                rules='同一位车手的同一个模型，在五条赛道各连续完成 3 圈。最先全部通过者夺冠；同一步通过按五站累计用时决胜，仍相同则并列。')
