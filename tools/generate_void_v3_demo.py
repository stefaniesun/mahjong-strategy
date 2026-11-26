"""生成换三张后定缺 void_v3 实盘 20 局可视化对比演示页 (outputs/assistant/void_v3_demo_20.html)。

特点:
- 精选 20 局换后真正纠结的牌局: 4/4/5撞车(7)、4/4/5非撞车(4)、庄家4/5/5与4/4/6(4)、3张临界(5);
- 展示换三张去向(换给谁、换出3张、换回3张、是否撞车)与换后13/14张实拍牌面;
- 内置“盲测模式”(先凭牌感点选万/条/筒,再揭晓 void_v3 / void_v2 / 旧版S3 对比);
- 内置“上帝视角复盘”折叠区(可展开查看另外三家实际定缺、是否放出单行道、以及定缺瞬间真实供牌)。
"""
from __future__ import annotations

import base64
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from assistant.swap_v1 import split_suits
from assistant.swap_v2 import choose_swap_v2_indices, choose_void_v2_idx
from assistant.void_v3 import score_void_candidates_v3, void_v3_sampling_probs
from engine.actions import Action, ActionKind
from engine.game import Game
from engine.tiles import SUITS, Tile, tile_to_str
from policies.heuristics import choose_void_suit
from tools.run_void_eval import _classify_deal, _suit_bucket

SUIT_CN = {"W": "万", "T": "条", "B": "筒"}
DIR_CN = {1: "换给上家(逆时针)", -1: "换给下家(顺时针)", 2: "换给对家"}
# Seat mapping matches engine: giver = (0 + swap_dir) % 4, receiver = (0 - swap_dir) % 4
# In standard table order 0=南(我), 1=下家, 2=对家, 3=上家:
# swap_dir=1 -> recv=3(上家), giver=1(下家); swap_dir=-1 -> recv=1(下家), giver=3(上家); swap_dir=2 -> recv=2(对家)
SEAT_NAME = {1: "下家", 2: "对家", 3: "上家"}


def load_tile_images() -> dict[str, str]:
    tile_dir = ROOT / "tools" / "mahjong_tiles"
    suit_prefix = {"W": "Man", "T": "Sou", "B": "Pin"}
    out: dict[str, str] = {}
    for code, prefix in suit_prefix.items():
        for rank in range(1, 10):
            p = tile_dir / f"{prefix}{rank}.png"
            b64 = base64.b64encode(p.read_bytes()).decode("ascii")
            out[f"{rank}{code}"] = f"data:image/png;base64,{b64}"
    return out


def collect_20_cases() -> list[dict]:
    # Target quotas per bucket, prioritizing informative disagreements + clear examples
    quotas = {
        "4/4/5_crash": (7, 4),      # (total, min_diff_from_v2_or_s3)
        "4/4/5_noncrash": (4, 2),
        "4/5/5_dealer": (4, 2),
        "3_close": (5, 3),
    }
    diff_buckets: dict[str, list[dict]] = {k: [] for k in quotas}
    same_buckets: dict[str, list[dict]] = {k: [] for k in quotas}

    for seed in range(30001, 45000):
        if all(
            len(diff_buckets[k]) >= quotas[k][1]
            and len(diff_buckets[k]) + len(same_buckets[k]) >= quotas[k][0]
            for k in quotas
        ):
            break
        dealer = seed % 4
        game = Game(seed=seed, dealer=dealer)
        state = game.reset()

        for p in range(4):
            idxs = choose_swap_v2_indices(
                state.hands[p].counts,
                is_dealer=(p == dealer),
                swap_direction=state.swap_direction,
                exact_live=False,
            )
            tiles = tuple(sorted(Tile.from_index(i) for i in idxs))
            game.step(p, Action(ActionKind.SWAP_THREE, tiles=tiles))

        p0_out_tiles = state.swap_choices[0]
        p0_out_suit = SUITS.index(p0_out_tiles[0].suit)
        giver_seat = (0 + state.swap_direction) % 4
        recv_seat = (0 - state.swap_direction) % 4
        p0_in_tiles = state.swap_choices[giver_seat]
        p0_in_suit = SUITS.index(p0_in_tiles[0].suit)
        is_crash = p0_out_suit == p0_in_suit

        bucket = _classify_deal(state.hands[0].counts, is_dealer=(dealer == 0), is_crash=is_crash)
        if bucket is None or bucket not in quotas:
            continue

        total_q, min_diff_q = quotas[bucket]
        if len(diff_buckets[bucket]) >= min_diff_q and len(diff_buckets[bucket]) + len(same_buckets[bucket]) >= total_q:
            continue

        p0_s3 = split_suits(state.hands[0].counts)
        v3_rows = score_void_candidates_v3(
            state.hands[0].counts,
            is_dealer=(dealer == 0),
            swap_out_suit=p0_out_suit,
            swap_in_suit=p0_in_suit,
            swap_direction=state.swap_direction,
        )
        v3_idx = v3_rows[0]["suit_idx"]
        v2_idx = choose_void_v2_idx(p0_s3, is_dealer=(dealer == 0))
        s3_idx = SUITS.index(choose_void_suit(state.hands[0]))

        # Check if interesting
        is_diff = (v3_idx != v2_idx) or (v3_idx != s3_idx)
        if is_diff and len(diff_buckets[bucket]) >= total_q:
            continue
        if (not is_diff) and len(same_buckets[bucket]) >= (total_q - min_diff_q):
            continue

        # Compute opponent real voids (using void_v3 with their own swap context!)
        opp_voids: dict[int, int] = {}
        for p in (1, 2, 3):
            p_out = SUITS.index(state.swap_choices[p][0].suit)
            p_giver = (p + state.swap_direction) % 4
            p_in = SUITS.index(state.swap_choices[p_giver][0].suit)
            opp_voids[p] = score_void_candidates_v3(
                state.hands[p].counts,
                is_dealer=(p == dealer),
                swap_out_suit=p_out,
                swap_in_suit=p_in,
                swap_direction=state.swap_direction,
            )[0]["suit_idx"]

        ov_list = [opp_voids[1], opp_voids[2], opp_voids[3]]
        wall_s = [0, 0, 0]
        for t in state.wall:
            wall_s[SUITS.index(t.suit)] += 1
        opp_hand_s = {
            p: [sum(state.hands[p].counts[s * 9 : (s + 1) * 9]) for s in range(3)]
            for p in (1, 2, 3)
        }
        other_pairs = ((1, 2), (0, 2), (0, 1))

        # Enrich each v3 option with true table metrics (hidden in spoiler until clicked)
        opts_enriched = []
        for row in v3_rows:
            vi = row["suit_idx"]
            kept = other_pairs[vi]
            free_feed = sum(opp_hand_s[p][s] for p in (1, 2, 3) for s in kept if opp_voids[p] == s)
            wall_avail = sum(wall_s[s] for s in kept)
            opp_same = ov_list.count(vi)
            sole_opp = opp_same == 2
            opts_enriched.append(
                {
                    **row,
                    "free_feed": free_feed,
                    "wall_avail": wall_avail,
                    "usable_supply": wall_avail + free_feed,
                    "opp_same_void": opp_same,
                    "sole_opp_keeper": sole_opp,
                }
            )

        hand_codes = [tile_to_str(t) for t in state.hands[0].tiles()]
        s_probs = void_v3_sampling_probs(v3_rows)
        if len(s_probs) == 1:
            sample_text = f"锁定缺{v3_rows[0]['suit']} 100%（优势明显不随机）"
        else:
            sample_text = " / ".join(
                f"缺{v3_rows[i]['suit']} {s_probs[i]:.0%}" for i in range(len(s_probs))
            )

        # 生成直白判定依据
        o0, o1 = v3_rows[0], v3_rows[1]
        if o0["shanten"] < o1["shanten"]:
            reason_text = f"第一层胜出：缺{o0['suit']}向听更快（{o0['shanten']}向听 vs 缺{o1['suit']}的{o1['shanten']}向听）"
        elif o0["void_cnt"] < o1["void_cnt"]:
            reason_text = f"缺门包袱更轻：缺{o0['suit']}仅{o0['void_cnt']}张（对比缺{o1['suit']}需打{o1['void_cnt']}张）"
        elif o0["ctx_cost"] != o1["ctx_cost"] and o0["base_cost"] >= o1["base_cost"]:
            reason_text = f"第三层拍板：两门牌面打平（同{o0['shanten']}向听、进张{o0['live']} vs {o1['live']}），由换牌流向防单行道定夺"
        elif o0["live"] > o1["live"]:
            reason_text = f"第二层胜出：同向听同张数，缺{o0['suit']}留牌有效进张更宽（{o0['live']}张 vs {o1['live']}张）"
        else:
            reason_text = f"综合牌形更优（含暗刻/连张与换牌流向微调）"

        case = {
            "seed": seed,
            "bucket": bucket,
            "shape_bucket": _suit_bucket(state.hands[0].counts),
            "is_dealer": dealer == 0,
            "swap_dir": state.swap_direction,
            "swap_dir_text": DIR_CN[state.swap_direction],
            "recv_name": SEAT_NAME[recv_seat],
            "giver_name": SEAT_NAME[giver_seat],
            "is_crash": is_crash,
            "out_tiles": [tile_to_str(t) for t in p0_out_tiles],
            "in_tiles": [tile_to_str(t) for t in p0_in_tiles],
            "out_suit_code": SUITS[p0_out_suit].value,
            "in_suit_code": SUITS[p0_in_suit].value,
            "hand": hand_codes,
            "v3_code": SUITS[v3_idx].value,
            "v2_code": SUITS[v2_idx].value,
            "s3_code": SUITS[s3_idx].value,
            "diff_from_v2": v3_idx != v2_idx,
            "diff_from_s3": v3_idx != s3_idx,
            "diff_any": is_diff,
            "reason_text": reason_text,
            "sample_text": sample_text,
            "opp_voids_text": {SEAT_NAME[p]: SUIT_CN[SUITS[opp_voids[p]].value] for p in (1, 2, 3)},
            "options": opts_enriched,
        }
        (diff_buckets[bucket] if is_diff else same_buckets[bucket]).append(case)

    ordered: list[dict] = []
    for k in ("4/4/5_crash", "4/4/5_noncrash", "4/5/5_dealer", "3_close"):
        total_q, _ = quotas[k]
        picked = diff_buckets[k][:total_q]
        rem = total_q - len(picked)
        if rem > 0:
            picked += same_buckets[k][:rem]
        ordered.extend(picked)

    for i, c in enumerate(ordered, start=1):
        c["case_no"] = i
    return ordered


HTML_TEMPLATE = """<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>换后定缺 void_v3 实盘 20 局对比演示</title>
<style>
  :root {
    --bg: #0f1318;
    --felt: #175c3d;
    --felt-hi: #217850;
    --panel: #f7f3e9;
    --ink: #26221b;
    --muted: #6f685a;
    --gold: #e7c266;
    --accent: #d94e34;
    --blue: #2b65c5;
    --green: #1e824c;
    --warn: #c46a10;
  }
  * { box-sizing: border-box; }
  body {
    margin: 0;
    background: radial-gradient(circle at 30% 10%, #1d2430, var(--bg) 70%);
    color: #f0ede4;
    font-family: "Microsoft YaHei", "PingFang SC", sans-serif;
    min-height: 100vh;
  }
  .topbar {
    position: sticky; top: 0; z-index: 20;
    background: rgba(19, 23, 31, 0.96);
    border-bottom: 1px solid #2c3342;
    padding: 10px 18px;
    display: flex; flex-wrap: wrap; gap: 10px; align-items: center;
    backdrop-filter: blur(8px);
  }
  .brand { font-size: 18px; font-weight: 700; color: var(--gold); letter-spacing: 0.5px; }
  .brand small { color: #a8a293; font-weight: 400; font-size: 12px; margin-left: 8px; }
  .filters { display: flex; gap: 6px; flex-wrap: wrap; }
  .btn {
    border: 1px solid #394154; background: #1e2430; color: #ddd;
    border-radius: 8px; padding: 5px 11px; cursor: pointer; font-size: 13px;
    transition: all .15s;
  }
  .btn:hover { border-color: var(--gold); color: #fff; }
  .btn.active { background: var(--gold); color: #241904; border-color: var(--gold); font-weight: 700; }
  .nav { margin-left: auto; display: flex; align-items: center; gap: 8px; }
  .pos { font-size: 13px; color: #ccc; min-width: 72px; text-align: center; }

  .strip {
    display: flex; gap: 6px; overflow-x: auto; padding: 10px 18px;
    background: rgba(0,0,0,.25); border-bottom: 1px solid #242a37;
  }
  .pill {
    flex: 0 0 auto; border: 1px solid #32394a; background: #191e28;
    border-radius: 10px; padding: 5px 10px; cursor: pointer; font-size: 12px; color: #c7c1b2;
    display: flex; align-items: center; gap: 5px;
  }
  .pill:hover { border-color: var(--gold); }
  .pill.active { background: #2c3446; border-color: var(--gold); color: #fff; box-shadow: 0 0 0 1px var(--gold) inset; }
  .dot-diff { width: 7px; height: 7px; border-radius: 50%; background: var(--accent); display: inline-block; }
  .dot-crash { width: 7px; height: 7px; border-radius: 50%; background: var(--gold); display: inline-block; }

  .container { max-width: 1240px; margin: 16px auto; padding: 0 16px 36px; display: grid; gap: 16px; }

  .table-card {
    background: linear-gradient(145deg, #623a1d, #442712);
    padding: 12px; border-radius: 20px; box-shadow: 0 16px 40px rgba(0,0,0,.55);
  }
  .felt {
    background: radial-gradient(ellipse at 50% 45%, var(--felt-hi), var(--felt) 70%, #10422b);
    border-radius: 13px; padding: 18px 20px 20px;
    border: 1.5px solid rgba(231, 194, 102, 0.35);
  }
  .meta-row { display: flex; flex-wrap: wrap; gap: 8px; align-items: center; margin-bottom: 12px; }
  .badge {
    display: inline-flex; align-items: center; gap: 4px;
    padding: 3px 10px; border-radius: 14px; font-size: 12px; font-weight: 600;
    background: rgba(0,0,0,.32); border: 1px solid rgba(255,255,255,.16); color: #f6eedc;
  }
  .badge.dealer { background: #b53421; border-color: #ff8e7a; }
  .badge.crash { background: rgba(217,78,52,.35); border-color: #ff8b74; color: #ffe0d8; }
  .badge.noncrash { background: rgba(43,101,197,.32); border-color: #84b4ff; color: #ddeaff; }
  .badge.bucket { background: rgba(231,194,102,.18); border-color: var(--gold); color: var(--gold); }
  .badge.diff { background: rgba(231,194,102,.28); border-color: var(--gold); color: #fff4cc; }

  .swap-flow {
    display: flex; flex-wrap: wrap; gap: 14px; align-items: center;
    background: rgba(0,0,0,.22); border: 1px solid rgba(255,255,255,.12);
    border-radius: 12px; padding: 10px 14px; margin-bottom: 14px;
  }
  .swap-part { display: flex; align-items: center; gap: 8px; font-size: 13px; color: #e7efe9; }
  .swap-arrow { color: var(--gold); font-weight: 700; font-size: 16px; }

  .hand-label {
    font-size: 13px; color: #d8eadf; margin-bottom: 8px;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
  }
  .hand-row {
    display: flex; flex-wrap: nowrap; align-items: flex-end; gap: 4px;
    padding: 20px 12px 10px; overflow-x: auto;
    background: rgba(0,0,0,.18); border-radius: 12px;
  }
  .suit-gap { width: 16px; flex: 0 0 auto; }

  .tile {
    position: relative; width: 58px; height: 84px; flex: 0 0 auto;
    border-radius: 8px;
    background: linear-gradient(180deg, #fffdf6 0%, #f3ebd6 88%, #dfd1b0 100%);
    border: 1px solid #cbb98f;
    box-shadow: 0 5px 0 #1f6d4a, 0 7px 10px rgba(0,0,0,.45);
    display: flex; align-items: center; justify-content: center;
    padding: 4px 3px 5px; transition: transform .16s ease, opacity .16s ease;
  }
  .tile img { width: 100%; height: 100%; object-fit: contain; pointer-events: none; }
  .tile.void-picked {
    transform: translateY(-10px);
    border: 2px solid var(--accent);
    box-shadow: 0 6px 0 #8f2d1b, 0 12px 18px rgba(217,78,52,.45);
    opacity: 0.88;
  }
  .tile.void-picked::after {
    content: "缺";
    position: absolute; top: -11px; left: 50%; transform: translateX(-50%);
    background: var(--accent); color: #fff; font-size: 10px; font-weight: 700;
    padding: 1px 6px; border-radius: 8px;
  }
  .tile.preview-void {
    transform: translateY(-10px);
    border: 2px solid var(--gold);
    box-shadow: 0 6px 0 #8a6414, 0 12px 18px rgba(231,194,102,.45);
  }
  .tile.preview-void::after {
    content: "预览缺";
    position: absolute; top: -11px; left: 50%; transform: translateX(-50%);
    background: var(--gold); color: #241904; font-size: 10px; font-weight: 700;
    padding: 1px 6px; border-radius: 8px; white-space: nowrap;
  }
  .tile.mini {
    width: 34px; height: 48px; border-radius: 5px; padding: 2px;
    box-shadow: 0 3px 0 #1f6d4a, 0 4px 6px rgba(0,0,0,.35);
  }

  .compare-strip {
    margin-top: 14px; display: grid; grid-template-columns: repeat(auto-fit, minmax(220px, 1fr)); gap: 10px;
  }
  .comp-box {
    background: rgba(0,0,0,.26); border: 1px solid rgba(255,255,255,.14);
    border-radius: 12px; padding: 10px 14px;
  }
  .comp-box.primary { border-color: rgba(231,194,102,.7); background: rgba(231,194,102,.12); }
  .comp-box.diffbox { border-color: rgba(217,78,52,.65); background: rgba(217,78,52,.12); }
  .comp-title { font-size: 12px; color: #d5e2da; display: flex; justify-content: space-between; margin-bottom: 6px; }
  .comp-big { font-size: 20px; font-weight: 700; color: #fff; }
  .comp-sub { font-size: 12px; color: #c9d8cf; margin-top: 4px; }

  .blind-choices { display: flex; gap: 10px; flex-wrap: wrap; align-items: center; }
  .suit-btn {
    border: 1.5px solid var(--gold); background: rgba(0,0,0,.35); color: #fff;
    border-radius: 10px; padding: 8px 20px; font-size: 15px; font-weight: 700; cursor: pointer;
  }
  .suit-btn:hover { background: var(--gold); color: #241904; }

  .panel {
    background: var(--panel); color: var(--ink);
    border-radius: 14px; padding: 16px 18px;
    box-shadow: 0 10px 26px rgba(0,0,0,.35);
  }
  .panel h3 {
    margin: 0 0 10px; font-size: 15px; color: #1b2430;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 8px;
  }
  table { width: 100%; border-collapse: collapse; font-size: 13px; }
  th, td { padding: 9px 10px; text-align: left; border-bottom: 1px solid #e3dccb; }
  th { color: var(--muted); font-weight: 600; font-size: 12px; }
  tr.opt-row { cursor: pointer; transition: background .12s; }
  tr.opt-row:hover { background: #efe8d8; }
  tr.opt-row.top-row { background: #e9f5ee; font-weight: 600; }
  .tag {
    display: inline-block; padding: 2px 7px; border-radius: 6px; font-size: 11px; font-weight: 700;
  }
  .tag.best { background: #1e824c; color: #fff; }
  .tag.v2 { background: #2b65c5; color: #fff; }
  .tag.s3 { background: #7f6b4b; color: #fff; }
  .tag.warn { background: #d94e34; color: #fff; }
  .tag.ok { background: #2d7d46; color: #fff; }

  .spoiler-bar {
    margin-top: 12px; padding-top: 12px; border-top: 1px dashed #d3cbb8;
    display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 10px;
  }
  .spoiler-btn {
    border: 1px solid #9c8f76; background: #ede6d5; color: #2b2418;
    border-radius: 8px; padding: 6px 12px; cursor: pointer; font-size: 13px; font-weight: 600;
  }
  .spoiler-btn:hover { background: #e2d7be; }
  .spoiler-box {
    margin-top: 10px; padding: 12px 14px; border-radius: 10px;
    background: #efe9da; border: 1px solid #d5cab0; font-size: 13px;
  }
</style>
</head>
<body>
<header class="topbar">
  <div class="brand">四川麻将 · 换后定缺 void_v3 对比演示 <small>20 局接近牌型 · 含4/4/5撞车与3张临界</small></div>
  <div class="filters">
    <button class="btn active" data-filter="all">全部 (20)</button>
    <button class="btn" data-filter="diff">v3 与旧版不同</button>
    <button class="btn" data-filter="4/4/5_crash">4/4/5 撞车</button>
    <button class="btn" data-filter="3_close">3张临界</button>
    <button class="btn" id="blindBtn">🙈 盲测模式：关</button>
  </div>
  <div class="nav">
    <button class="btn" id="prevBtn">◀ 上一局</button>
    <span class="pos" id="posText">1 / 20</span>
    <button class="btn" id="nextBtn">下一局 ▶</button>
  </div>
</header>

<div class="strip" id="strip"></div>
<main class="container" id="main"></main>

<script>
const TILE_IMG = __TILE_IMG__;
const CASES = __CASES__;
const SUIT_CN = {W: "万", T: "条", B: "筒"};
const SUIT_ORDER = {W: 0, T: 1, B: 2};
const BUCKET_CN = {
  "4/4/5_crash": "4/4/5 撞车",
  "4/4/5_noncrash": "4/4/5 非撞车",
  "4/5/5_dealer": "庄家 4/5/5·4/4/6",
  "3_close": "3张临界"
};

let filterMode = "all";
let visibleIndices = CASES.map((_, i) => i);
let curPos = 0;
let blindMode = false;
let blindRevealed = true;
let userChoice = null;
let previewSuit = null;
let spoilerOpen = false;

function tileEl(code, cls = "") {
  return `<div class="tile ${cls}"><img src="${TILE_IMG[code]}" alt="${code}" title="${code}"></div>`;
}

function sortCodes(list) {
  return [...list].sort((a, b) => SUIT_ORDER[a.slice(-1)] * 10 + Number(a.slice(0, -1)) - (SUIT_ORDER[b.slice(-1)] * 10 + Number(b.slice(0, -1))));
}

function resetView() {
  previewSuit = null;
  spoilerOpen = false;
  userChoice = null;
  blindRevealed = !blindMode;
}

function applyFilter(mode) {
  filterMode = mode;
  document.querySelectorAll('.filters .btn[data-filter]').forEach(b => b.classList.toggle('active', b.dataset.filter === mode));
  visibleIndices = CASES.map((c, i) => ({c, i}))
    .filter(({c}) => mode === "all" ? true : (mode === "diff" ? c.diff_any : c.bucket === mode))
    .map(({i}) => i);
  curPos = 0;
  resetView();
  renderAll();
}

function go(delta) {
  if (!visibleIndices.length) return;
  curPos = (curPos + delta + visibleIndices.length) % visibleIndices.length;
  resetView();
  renderAll();
}

function jumpTo(caseIdx) {
  const p = visibleIndices.indexOf(caseIdx);
  if (p >= 0) {
    curPos = p;
    resetView();
    renderAll();
  }
}

function toggleBlind() {
  blindMode = !blindMode;
  const btn = document.getElementById('blindBtn');
  btn.textContent = blindMode ? '👁️ 盲测模式：开' : '🙈 盲测模式：关';
  btn.classList.toggle('active', blindMode);
  resetView();
  renderAll();
}

function pickBlindSuit(code) {
  userChoice = code;
  blindRevealed = true;
  renderAll();
}

function togglePreview(code) {
  if (!blindRevealed) return;
  previewSuit = (previewSuit === code) ? null : code;
  renderAll();
}

function toggleSpoiler() {
  spoilerOpen = !spoilerOpen;
  renderAll();
}

function renderStrip() {
  const html = visibleIndices.map((idx, p) => {
    const c = CASES[idx];
    const dots = (c.diff_any ? '<span class="dot-diff" title="v3与旧版不同"></span>' : '') +
                 (c.is_crash ? '<span class="dot-crash" title="换牌撞车"></span>' : '');
    return `<button class="pill ${p === curPos ? 'active' : ''}" onclick="jumpTo(${idx})">
      #${c.case_no} · ${BUCKET_CN[c.bucket]} ${dots}
    </button>`;
  }).join('');
  document.getElementById('strip').innerHTML = html || '<span style="color:#aaa;font-size:12px">无匹配牌局</span>';
}

function renderMain() {
  const mainEl = document.getElementById('main');
  if (!visibleIndices.length) {
    mainEl.innerHTML = '<div class="panel">当前筛选条件下没有牌局。</div>';
    document.getElementById('posText').textContent = '0 / 0';
    return;
  }
  const c = CASES[visibleIndices[curPos]];
  document.getElementById('posText').textContent = `${curPos + 1} / ${visibleIndices.length}`;

  const hand = sortCodes(c.hand);
  const activeVoid = blindRevealed ? (previewSuit || c.v3_code) : null;
  const isPreview = !!previewSuit;

  let handHtml = '';
  for (let i = 0; i < hand.length; i++) {
    if (i > 0 && hand[i].slice(-1) !== hand[i - 1].slice(-1)) {
      handHtml += '<span class="suit-gap"></span>';
    }
    let cls = '';
    if (activeVoid && hand[i].slice(-1) === activeVoid) {
      cls = isPreview ? 'preview-void' : 'void-picked';
    }
    handHtml += tileEl(hand[i], cls);
  }

  const top = c.options[0];
  let compareHtml = '';
  if (!blindRevealed) {
    compareHtml = `<div class="compare-strip">
      <div class="comp-box primary">
        <div class="comp-title"><b>盲测中：请根据手牌与上方换牌流向，选出你的定缺花色</b><span>点击即揭晓对比</span></div>
        <div class="blind-choices">
          <button class="suit-btn" onclick="pickBlindSuit('W')">缺 万</button>
          <button class="suit-btn" onclick="pickBlindSuit('T')">缺 条</button>
          <button class="suit-btn" onclick="pickBlindSuit('B')">缺 筒</button>
        </div>
      </div>
    </div>`;
  } else {
    const userBox = userChoice
      ? `<div class="comp-box"><div class="comp-title"><b>你的直觉定缺</b><span>${userChoice === c.v3_code ? '✅ 与 void_v3 一致' : '💡 可对比下方得失'}</span></div><div class="comp-big">缺 ${SUIT_CN[userChoice]}</div></div>`
      : '';
    compareHtml = `<div class="compare-strip">
      <div class="comp-box primary">
        <div class="comp-title"><b>void_v3 推荐定缺</b><span>综合代价 ${top.total_cost.toFixed(2)}</span></div>
        <div class="comp-big">缺 ${SUIT_CN[c.v3_code]} <span style="font-size:13px;font-weight:400;color:#e7c266">(${top.void_cnt}张 · 留牌${top.shanten}向听 · 有效进张${top.live}张)</span></div>
        <div class="comp-sub">💡 ${c.reason_text}</div>
        <div class="comp-sub" style="color:#b7d9c4">🎲 AI训练抽样：${c.sample_text}</div>
      </div>
      ${userBox}
      <div class="comp-box ${c.diff_from_v2 ? 'diffbox' : ''}">
        <div class="comp-title"><b>旧版 void_v2</b><span>${c.diff_from_v2 ? '⚠️ 与 v3 不同' : '与 v3 一致'}</span></div>
        <div class="comp-big">缺 ${SUIT_CN[c.v2_code]}</div>
        <div class="comp-sub">未计入主进张权重与换牌流向</div>
      </div>
      <div class="comp-box ${c.diff_from_s3 ? 'diffbox' : ''}">
        <div class="comp-title"><b>旧版 S3 规则</b><span>${c.diff_from_s3 ? '⚠️ 与 v3 不同' : '与 v3 一致'}</span></div>
        <div class="comp-big">缺 ${SUIT_CN[c.s3_code]}</div>
        <div class="comp-sub">仅按最少张数/死规则定缺</div>
      </div>
    </div>`;
  }

  let tableHtml = '';
  if (blindRevealed) {
    const rowsHtml = c.options.map((o, idx) => {
      const tags = [
        idx === 0 ? '<span class="tag best">v3 首选</span>' : '',
        o.suit_code === c.v2_code ? '<span class="tag v2">v2</span>' : '',
        o.suit_code === c.s3_code ? '<span class="tag s3">S3</span>' : '',
      ].filter(Boolean).join(' ');
      const ctxStr = o.ctx_cost !== 0 ? `${o.ctx_cost > 0 ? '+' : ''}${o.ctx_cost.toFixed(2)}` : '0.00';
      return `<tr class="opt-row ${idx === 0 ? 'top-row' : ''}" onclick="togglePreview('${o.suit_code}')">
        <td><b>缺${o.suit}</b> ${tags}</td>
        <td>${o.void_cnt} 张</td>
        <td>${o.shanten} 向听</td>
        <td><b>${o.live} 张</b></td>
        <td>${o.kept_pairs} 对 / ${o.kept_trips} 刻</td>
        <td>${o.base_cost.toFixed(2)}</td>
        <td>${ctxStr}</td>
        <td><b>${o.total_cost.toFixed(2)}</b></td>
        <td>${o.ctx_note || '—'}</td>
      </tr>`;
    }).join('');

    const spoilerRows = c.options.map(o => {
      const laneTag = o.sole_opp_keeper
        ? '<span class="tag warn">⚠️ 放对面单行道(3缺1)</span>'
        : (o.opp_same_void === 1 ? '<span class="tag ok">2+1+1 均衡</span>' : '<span class="tag s3">独缺(需防喂3家)</span>');
      return `<tr>
        <td><b>缺${o.suit}</b></td>
        <td>${o.opp_same_void} 家对手同缺</td>
        <td>${laneTag}</td>
        <td>牌墙 ${o.wall_avail} + 对手必打喂牌 ${o.free_feed} = <b>${o.usable_supply} 张</b></td>
      </tr>`;
    }).join('');

    const spoilerSection = spoilerOpen ? `<div class="spoiler-box">
      <div style="margin-bottom:8px"><b>三家对手实际定缺：</b>下家缺<b>${c.opp_voids_text["下家"]}</b> · 对家缺<b>${c.opp_voids_text["对家"]}</b> · 上家缺<b>${c.opp_voids_text["上家"]}</b></div>
      <table>
        <thead><tr><th>若你定缺</th><th>撞缺对手数</th><th>牌桌格局</th><th>定缺瞬间可用总供牌</th></tr></thead>
        <tbody>${spoilerRows}</tbody>
      </table>
    </div>` : '';

    tableHtml = `<section class="panel">
      <h3>三门定缺详细对比 <span style="font-size:12px;font-weight:400;color:var(--muted)">点击任意一行可在牌桌上预览该门定缺</span></h3>
      <table>
        <thead>
          <tr><th>定缺花色</th><th>缺门张数</th><th>缺后向听</th><th>有效进张</th><th>留牌对/刻</th><th>牌形代价</th><th>流向修正</th><th>总代价</th><th>换牌流向备注</th></tr>
        </thead>
        <tbody>${rowsHtml}</tbody>
      </table>
      <div class="spoiler-bar">
        <span style="font-size:12px;color:var(--muted)">先用实战视角判断完，再点右侧按钮核对四家真实暗牌与单行道格局</span>
        <button class="spoiler-btn" onclick="toggleSpoiler()">${spoilerOpen ? '🔼 收起上帝视角复盘' : '🔍 展开上帝视角复盘（看真实单行道与供牌）'}</button>
      </div>
      ${spoilerSection}
    </section>`;
  }

  mainEl.innerHTML = `
    <section class="table-card">
      <div class="felt">
        <div class="meta-row">
          <span class="badge ${c.is_dealer ? 'dealer' : ''}">${c.is_dealer ? '庄家 · 14张' : '闲家 · 13张'}</span>
          <span class="badge bucket">${BUCKET_CN[c.bucket]} (${c.shape_bucket})</span>
          <span class="badge ${c.is_crash ? 'crash' : 'noncrash'}">${c.is_crash ? '💥 换牌撞车（出' + SUIT_CN[c.out_suit_code] + '回' + SUIT_CN[c.in_suit_code] + '）' : '非撞车（出' + SUIT_CN[c.out_suit_code] + '回' + SUIT_CN[c.in_suit_code] + '）'}</span>
          <span class="badge">${c.swap_dir_text}</span>
          ${c.diff_any ? '<span class="badge diff">与旧版策略不同</span>' : ''}
          <span class="badge" style="margin-left:auto">Seed #${c.seed}</span>
        </div>

        <div class="swap-flow">
          <div class="swap-part">
            <span>换给<b>${c.recv_name}</b>：</span>
            ${sortCodes(c.out_tiles).map(x => tileEl(x, 'mini')).join('')}
          </div>
          <span class="swap-arrow">⇄</span>
          <div class="swap-part">
            <span>从<b>${c.giver_name}</b>换回：</span>
            ${sortCodes(c.in_tiles).map(x => tileEl(x, 'mini')).join('')}
          </div>
        </div>

        <div class="hand-label">
          <span>换后当前手牌（按 万 / 条 / 筒 排序）</span>
          <span>${blindRevealed ? (isPreview ? '正在预览：缺' + SUIT_CN[previewSuit] + '（再次点击取消）' : '高亮红标为 void_v3 推荐定缺') : '盲测模式已开启'}</span>
        </div>
        <div class="hand-row">${handHtml}</div>
        ${compareHtml}
      </div>
    </section>
    ${tableHtml}
  `;
}

function renderAll() {
  renderStrip();
  renderMain();
}

document.querySelectorAll('.filters .btn[data-filter]').forEach(btn => {
  btn.addEventListener('click', () => applyFilter(btn.dataset.filter));
});
document.getElementById('blindBtn').addEventListener('click', toggleBlind);
document.getElementById('prevBtn').addEventListener('click', () => go(-1));
document.getElementById('nextBtn').addEventListener('click', () => go(1));
document.addEventListener('keydown', e => {
  if (e.key === 'ArrowLeft') go(-1);
  if (e.key === 'ArrowRight') go(1);
});

renderAll();
</script>
</body>
</html>
"""


def main() -> None:
    out_path = ROOT / "outputs" / "assistant" / "void_v3_demo_20.html"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tile_imgs = load_tile_images()
    cases = collect_20_cases()
    html = HTML_TEMPLATE.replace("__TILE_IMG__", json.dumps(tile_imgs, ensure_ascii=False))
    html = html.replace("__CASES__", json.dumps(cases, ensure_ascii=False))
    out_path.write_text(html, encoding="utf-8")
    diff_v2 = sum(1 for c in cases if c["diff_from_v2"])
    diff_s3 = sum(1 for c in cases if c["diff_from_s3"])
    print(f"Generated {len(cases)} cases -> {out_path} (diff_from_v2={diff_v2}, diff_from_s3={diff_s3})")


if __name__ == "__main__":
    main()
