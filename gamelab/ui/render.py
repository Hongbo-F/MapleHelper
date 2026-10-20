"""Draws the mini-map: the map from the game data, and on it what the stack sees, predicts and plans
(gamelab.ui.state.State), with a text panel under it. Pure drawing: a State in, an RGB picture out, so
it's tested offline and the window only shows the result."""

import copy
import time

import cv2
import numpy as np

from .. import directions
from ..attack.reach import SHOT_DOWN, SHOT_UP, STAR_REACH, clear_span, shot_dx
from ..localization.mapdata import MapModel
from .state import State

BACKGROUND = (16, 20, 30)
WALL = (85, 85, 95)
PLATFORM = (70, 150, 80)
SAFE = (60, 170, 190)        # no monster can touch her there
GROUND = (150, 230, 110)     # a hunting ground of the tour
ROPE = (200, 140, 60)
PORTAL = (90, 150, 255)
TEXT = (210, 215, 225)
DIM = (120, 125, 140)
HER = (255, 225, 40)
ROUTE = (80, 220, 255)
CHAIN = (255, 90, 230)
DANGER = (230, 60, 60)
LOOT = (255, 210, 60)
DROPS = (255, 150, 40)
HER_TRAIL = (150, 130, 40)
UNSEEN = (255, 80, 255)      # her name tag hidden: placed from her minimap dot
REACH_FACING = (90, 90, 150)  # where her stars reach, on the side she faces
REACH_BEHIND = (55, 55, 85)
MOB_HP = (90, 230, 90)
HP_BAR, MP_BAR = (230, 70, 70), (80, 130, 255)
BAR_BACK = (50, 55, 70)      # empty bars, and the line above the text panel
KIND = {"pig": (255, 150, 170), "slime": (110, 235, 110), "snail": (130, 170, 255),
        "orange-mushroom": (255, 150, 40)}
FONT = cv2.FONT_HERSHEY_SIMPLEX
JUMP_ARC = 77                # how high a jump's arc is drawn, in map units (physics.JUMP_HEIGHT)
REACH_ALPHA = 0.35           # how strongly her reach's bands show over the map
# The map's colours by name: the web page draws the same map in the browser, with these (web.map_payload).
STYLE = {"background": BACKGROUND, "wall": WALL, "platform": PLATFORM, "safe": SAFE, "ground": GROUND,
         "rope": ROPE, "portal": PORTAL, "text": TEXT, "dim": DIM, "her": HER, "route": ROUTE, "chain": CHAIN,
         "danger": DANGER, "loot": LOOT, "drops": DROPS, "her_trail": HER_TRAIL, "unseen": UNSEEN,
         "reach_facing": REACH_FACING, "reach_behind": REACH_BEHIND, "mob_hp": MOB_HP}


class MapView:
    """The map fitted into a picture `width` px wide, with a text panel `panel` px tall under it."""

    def __init__(self, model: MapModel, width: int = 760, panel: int = 170, margin: int = 14):
        self.model = model
        xs = [p.left for p in model.platforms] + [p.right for p in model.platforms] + [r.x for r in model.ropes]
        ys = [p.top for p in model.platforms] + [s.y1 for p in model.platforms for s in p.segments] \
            + [r.top for r in model.ropes] + [r.bottom for r in model.ropes]
        self.x0, self.x1 = min(xs) - 30, max(xs) + 30
        self.y0, self.y1 = min(ys) - 110, max(ys) + 30
        self.scale = (width - 2 * margin) / (self.x1 - self.x0)
        self.margin = margin
        self.map_h = int((self.y1 - self.y0) * self.scale) + 2 * margin
        self.size = (width, self.map_h + panel)
        self._left = self._top = 0  # where the picture drawn on starts (draw(crop=...): a part of the map)
        self._base = self._draw_base()
        self._roles: tuple | None = None   # the tour's (safe, grounds) drawn into _with_roles
        self._with_roles = self._base

    def pt(self, x: float, y: float) -> tuple[int, int]:
        """Picture px of map point (x, y)."""
        return (int(round((x - self.x0) * self.scale)) + self.margin - self._left,
                int(round((y - self.y0) * self.scale)) + self.margin - self._top)

    def draw(self, state: State, crop: tuple[int, int, int, int] | None = None) -> np.ndarray:
        """The picture of `state`: the map with the text panel under it; or, with `crop` (left, top, width,
        height in picture px), just that part of the map, drawn only there: /frame.jpg's "around her" is
        900 px of a 2250-px map, which took 11 ms drawn whole and cut, and takes 1.5-2 ms. It's the same
        as draw(state)[top:top + height, left:left + width] but for the antialiasing of lines where they
        cross its edges (OpenCV cuts them there): 11-27 px of its 648,000 differ, on five recorded runs."""
        tour = state.tour or {}
        base = self._roles_layer(tour)
        if crop is None:
            img = base.copy()
            self._layers(img[:self.map_h], state, tour)  # map drawings stay out of the text panel
            self._panel(img, state)
            return img
        left, top, w, h = crop
        h = min(h, self.map_h - top)
        img = base[top:top + h, left:left + w].copy()
        part = copy.copy(self)  # the same view, its points moved to the part's corner
        part._left, part._top = left, top
        part._layers(img, state, tour)
        return img

    def _layers(self, img, state: State, tour: dict) -> None:
        """Everything drawn over the map (the tour's platforms are in the base already)."""
        self._tour(img, tour)
        self._monsters(img, state.monsters)
        self._route(img, state.route)
        self._chain(img, state.chain)
        self._drops(img, state.drops)
        self._piles(img, state.hunt)
        scene = state.scene or {}
        her = state.her()
        if her is not None:
            self._reach(img, her)
        self._monsters_seen(img, scene.get("monsters") or [], state.stance)
        self._loot(img, scene.get("loot") or [])
        self._trail(img, state.trail)
        if her is not None:
            self._her(img, her)

    # --- the map ----------------------------------------------------------------------------------

    def _draw_base(self) -> np.ndarray:
        w, h = self.size
        full = np.full((h, w, 3), BACKGROUND, np.uint8)
        img = full[:self.map_h]
        m = self.model
        for wall in m.walls:
            cv2.line(img, self.pt(wall.x1, wall.y1), self.pt(wall.x2, wall.y2), WALL, 1, cv2.LINE_AA)
        for r in m.ropes:
            cv2.line(img, self.pt(r.x, r.top), self.pt(r.x, r.bottom), ROPE, 1, cv2.LINE_AA)
        for p in m.platforms:
            self._platform(img, p, PLATFORM, 2)
            x, y = self.pt(p.left, p.y_at(p.left))
            cv2.putText(img, str(p.id), (x, y - 4), FONT, 0.32, DIM, 1, cv2.LINE_AA)
        for portal in m.portals:
            cv2.circle(img, self.pt(portal.x, portal.y), 4, PORTAL, 1, cv2.LINE_AA)
        cv2.line(full, (0, self.map_h), (w, self.map_h), BAR_BACK, 1)
        return full

    def _known(self, pid) -> bool:
        """Whether this map has platform `pid` (a record may name one it doesn't)."""
        return 0 <= pid < len(self.model.platforms)

    def _platform(self, img, p, colour, thickness) -> None:
        for s in p.segments:
            cv2.line(img, self.pt(s.x1, s.y1), self.pt(s.x2, s.y2), colour, thickness, cv2.LINE_AA)

    def _roles_layer(self, tour: dict) -> np.ndarray:
        """The map with the tour's safe platforms and hunting grounds drawn in, kept until they change:
        drawn every time, they were a quarter of a 900-px picture's 3.6 ms."""
        key = (tuple(tour.get("safe") or ()), tuple(tour.get("grounds") or ()))
        if key != self._roles:
            img = self._base.copy() if any(key) else self._base
            self._platform_roles(img[:self.map_h], tour)
            self._roles, self._with_roles = key, img
        return self._with_roles

    def _platform_roles(self, img, tour: dict) -> None:
        for pid in tour.get("safe") or []:
            if self._known(pid):
                self._platform(img, self.model.platform(pid), SAFE, 2)
        for pid in tour.get("grounds") or []:
            if self._known(int(pid)):
                self._platform(img, self.model.platform(int(pid)), GROUND, 3)

    # --- what's planned ---------------------------------------------------------------------------

    def _tour(self, img, tour: dict) -> None:
        for i, pid in enumerate(tour.get("order") or []):
            if not self._known(pid):
                continue
            p = self.model.platform(pid)
            x, y = self.pt((p.left + p.right) / 2, p.y_at((p.left + p.right) / 2))
            current = pid == tour.get("stop")
            cv2.circle(img, (x, y + 12), 8, HER if current else GROUND, -1 if current else 1, cv2.LINE_AA)
            cv2.putText(img, str(i + 1), (x - 4, y + 16), FONT, 0.35, BACKGROUND if current else GROUND, 1, cv2.LINE_AA)
        spot = tour.get("spot")
        if spot and self._known(spot.get("platform", -1)):
            p = self.model.platform(spot["platform"])
            x, y = self.pt(spot["x"], p.y_at(spot["x"]))
            cv2.drawMarker(img, (x, y - 6), HER, cv2.MARKER_DIAMOND, 12, 2, cv2.LINE_AA)

    def _route(self, img, route: dict | None) -> None:
        if not route:
            return
        plats = self.model.platforms
        for i, mv in enumerate(route.get("moves") or []):
            if not (self._known(mv["src"]) and self._known(mv["dst"])):
                continue
            a = self.pt(mv["x"], plats[mv["src"]].y_at(mv["x"]))
            b = self.pt(mv["land_x"], plats[mv["dst"]].y_at(mv["land_x"]))
            if mv["kind"] in ("jump", "walk_off", "hop"):
                self._arc(img, a, b, ROUTE)
            else:
                cv2.arrowedLine(img, a, b, ROUTE, 1, cv2.LINE_AA, tipLength=0.15)
            if i == 0:
                cv2.circle(img, a, 3, ROUTE, -1, cv2.LINE_AA)

    def _arc(self, img, a, b, colour) -> None:
        """A jump's path: up from a, then down onto b."""
        lift = max(8, int(JUMP_ARC * self.scale))
        pts = []
        for i in range(13):
            f = i / 12
            x = a[0] + (b[0] - a[0]) * f
            y = a[1] + (b[1] - a[1]) * f - 4 * lift * f * (1 - f)
            pts.append((int(x), int(y)))
        cv2.polylines(img, [np.array(pts, np.int32)], False, colour, 1, cv2.LINE_AA)
        cv2.circle(img, b, 2, colour, -1, cv2.LINE_AA)

    def _chain(self, img, chain: dict | None) -> None:
        if not chain:
            return
        plats = self.model.platforms
        for leg in chain.get("legs") or []:
            for pid, x0, x1 in leg.get("spans") or []:
                if self._known(pid):
                    p = plats[pid]
                    cv2.line(img, self.pt(x0, p.y_at(x0) - 6), self.pt(x1, p.y_at(x1) - 6), CHAIN, 3, cv2.LINE_AA)
            end = leg.get("end")
            if end and self._known(end.get("platform", -1)):
                p = plats[end["platform"]]
                cv2.circle(img, self.pt(end["x"], p.y_at(end["x"]) - 6), 3, CHAIN, -1, cv2.LINE_AA)

    # --- what's seen and predicted ----------------------------------------------------------------

    def _monsters(self, img, monsters: dict | None) -> None:
        """Where each monster can be over the next moment, as the planner predicts it."""
        if not monsters:
            return
        plats = self.model.platforms
        for mo in monsters.get("list") or []:
            pid, reach = mo.get("platform"), mo.get("reach")
            if reach and pid is not None and self._known(pid):
                p = plats[pid]
                y = p.y_at((reach[0] + reach[1]) / 2) - 3
                cv2.line(img, self.pt(reach[0], y), self.pt(reach[1], y), DANGER, 2, cv2.LINE_AA)

    def _monsters_seen(self, img, monsters: list, stance: dict | None) -> None:
        """The monsters in the last look, coloured by kind; the ones the last throw aimed at ringed."""
        aimed = {(round(a["x"]), round(a["y"])) for a in (stance or {}).get("aimed") or []}
        for t in monsters:
            x, y = self.pt(t["x"], t["y"] - 12)
            colour = KIND.get(t["kind"], TEXT)
            cv2.circle(img, (x, y), 4, colour, -1, cv2.LINE_AA)
            if (round(t["x"]), round(t["y"])) in aimed:
                cv2.circle(img, (x, y), 8, DANGER, 1, cv2.LINE_AA)
            if t.get("bar") is not None:
                cv2.line(img, (x - 6, y - 7), (x - 6 + int(12 * t["bar"]), y - 7), MOB_HP, 2)

    def _loot(self, img, loot: list) -> None:
        for d in loot:
            x, y = self.pt(d["x"], d["y"] - 4)
            cv2.rectangle(img, (x - 2, y - 2), (x + 2, y + 2), LOOT, -1)

    def _drops(self, img, drops: dict | None) -> None:
        for d in (drops or {}).get("pending") or []:
            if d.get("platform") is None:
                continue
            x, y = self.pt(d["x"], d["y"] - 4)
            cv2.rectangle(img, (x - 4, y - 4), (x + 4, y + 4), LOOT if d.get("mesos") else DROPS, 1)

    def _piles(self, img, hunt: dict | None) -> None:
        """The hunt loop's drop store: her piles bright, others' dim, expected ones (not seen yet) small,
        urgent ones red."""
        for d in (hunt or {}).get("piles") or []:
            if d.get("platform") is None or d.get("y") is None:
                continue
            x, y = self.pt(d["x"], d["y"] - 4)
            r = 3 if d.get("status") == "expected" else 5
            colour = DANGER if d.get("urgent") else (DROPS if d.get("hers") else DIM)
            cv2.rectangle(img, (x - r, y - r), (x + r, y + r), colour, 1)

    def _reach(self, img, her: dict) -> None:
        """Where a star from her reaches: a band on each side, the side she faces brighter, cut short by a
        wall of her platform's foothold group (attack.reach.clear_span: on 4, at the step up to 2).

        Blended into the rows the bands cover only: blending the whole picture (it comes out unchanged
        outside the bands) was a third of a 900-px picture's time and over half of a 2250-px one's. Whole
        rows and one more, as one block, so the result is the same to the last px: OpenCV's vector code
        blends all but a block's last few px, which other code does, at times 1 level apart (blending each
        band's own rectangle left a px at its right end 1 level off in 4 of 435 pictures)."""
        x, y = her["x"], her["y"]
        h = img.shape[0]
        top, bottom = self.pt(x, y - SHOT_UP)[1], self.pt(x, y + SHOT_DOWN)[1]
        if bottom < 0 or top >= h:
            return
        pid = her.get("platform")
        stops = clear_span(self.model.platform(pid), x, y) if pid is not None and self._known(pid) else None
        first = max(top, 0) if bottom < h - 1 else 0  # down to the last row: the picture's own last px
        rows = img[first:min(bottom + 2, h)]
        overlay = rows.copy()
        for sign in (-1, 1):
            near, far = x + sign * shot_dx(0), x + sign * STAR_REACH
            if stops is not None:
                far = max(far, stops[0]) if sign < 0 else min(far, stops[1])
                if (far - near) * sign <= 0:
                    continue
            (ax, ay), (bx, by) = self.pt(min(near, far), y - SHOT_UP), self.pt(max(near, far), y + SHOT_DOWN)
            facing = her.get("facing") == directions.toward(sign)
            cv2.rectangle(overlay, (ax, ay - first), (bx, by - first), REACH_FACING if facing else REACH_BEHIND, -1)
        cv2.addWeighted(overlay, REACH_ALPHA, rows, 1 - REACH_ALPHA, 0, rows)

    def _trail(self, img, trail) -> None:
        if len(trail) >= 2:
            pts = np.array([self.pt(x, y - 4) for x, y in trail], np.int32)
            cv2.polylines(img, [pts], False, HER_TRAIL, 1, cv2.LINE_AA)

    def _her(self, img, her: dict) -> None:
        x, y = self.pt(her["x"], her["y"] - 8)
        cv2.circle(img, (x, y), 6, HER, -1, cv2.LINE_AA)
        if her.get("facing") in ("left", "right"):
            d = directions.sign(her["facing"])
            cv2.arrowedLine(img, (x, y), (x + 14 * d, y), BACKGROUND, 2, cv2.LINE_AA, tipLength=0.5)
        if her.get("seen") is False:
            cv2.circle(img, (x, y), 9, UNSEEN, 1, cv2.LINE_AA)

    # --- text panel -------------------------------------------------------------------------------

    def _panel(self, img, state: State) -> None:
        top = self.map_h + 8
        scene = state.scene or {}
        her = state.her()
        x = 10
        for label, value, colour in (("HP", scene.get("hp"), HP_BAR), ("MP", scene.get("mp"), MP_BAR)):
            cv2.putText(img, label, (x, top + 12), FONT, 0.4, TEXT, 1, cv2.LINE_AA)
            cv2.rectangle(img, (x + 24, top + 3), (x + 124, top + 13), BAR_BACK, -1)
            if value is not None:
                cv2.rectangle(img, (x + 24, top + 3), (x + 24 + int(100 * value), top + 13), colour, -1)
            x += 140
        mode = state.session.get("mode", "live")
        when = time.strftime("%H:%M:%S", time.localtime(state.t)) if state.t else "--:--:--"
        where = "lost" if her is None else (f"x {her['x']:.0f}  platform {her.get('platform')}  ({her.get('how', '?')})")
        cv2.putText(img, f"{mode}  {when}   {where}", (x, top + 12), FONT, 0.4, TEXT, 1, cv2.LINE_AA)
        line = top + 30
        stance = state.stance
        if stance:
            text = stance.get("text") or stance.get("act", "")
            cv2.putText(img, f"fight: {text}", (10, line), FONT, 0.38, DANGER, 1, cv2.LINE_AA)
            line += 16
        if state.chain:
            cv2.putText(img, f"moving: {state.chain.get('text', '')}", (10, line), FONT, 0.38, CHAIN, 1, cv2.LINE_AA)
            line += 16
        hunt = state.hunt
        if hunt:  # the hunt loop's current rule and objective (behavior.hunt)
            what = hunt.get("objective") or f"platform {hunt.get('ground')}"
            rates = ""
            if hunt.get("rate_here") is not None:
                rates = f"  ({hunt['rate_here']:.2f} EXP/s here"
                best = hunt.get("rate_to_beat")
                rates += f", {best[0]:.2f} at platform {best[1]})" if best else ")"
            cv2.putText(img, f"hunt: {hunt.get('rule', '')} - {what}{rates}", (10, line), FONT, 0.38, TEXT, 1,
                        cv2.LINE_AA)
            line += 16
        for text in list(state.logs)[-max(0, (self.size[1] - line) // 15):]:
            cv2.putText(img, text[:110], (10, line), FONT, 0.36, DIM, 1, cv2.LINE_AA)
            line += 15
