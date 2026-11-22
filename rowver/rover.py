"""One rover in the shared world: pose, brake, wheel commands, drive segments, docking.

The rover only knows what a real rover knows about itself: its wheel and steering commands, its
pose from odometry/localisation (here the chassis pose plus noise), its IMU tilt. It does not
know the terrain physics; it measures what each drive segment actually did.

Two kinds of base: the holonomic LeKiwi (moves any direction) and the non-holonomic Mars rover
(world.ROVER = mars_rover.py: 4 wheels on rockers, skid steering: drives arcs and turns in place,
never sideways). The steering law and the docking routine below handle both.
"""
from __future__ import annotations

import mujoco
import numpy as np

from . import world as W


def wrap(a):
    return (np.asarray(a, float) + np.pi) % (2 * np.pi) - np.pi


class Rover:
    RAMP_S = 0.25        # s: a new wheel command is blended in over this time (no velocity steps)
    TURN_IN_PLACE = np.radians(50)     # steered base: a heading error above this turns in place first

    def __init__(self, world, rid: str, seed=0, pose_noise=0.004):
        self.w, self.rid = world, rid
        m, d = world.m, world.d
        self.ch = m.body(f"{rid}/{W.CHASSIS}").id
        self.design = world.designs[rid]
        self.kin = W.ROVER.kinematics(m, d, f"{rid}/", self.design)
        self.holonomic = bool(getattr(self.kin, "holonomic", True))
        j = m.body_jntadr[self.ch]
        self.qadr, self.vadr = m.jnt_qposadr[j], m.jnt_dofadr[j]
        self.rng = np.random.default_rng(seed)
        self.pose_noise = pose_noise
        self.braked = True
        self.carrying = False          # set by a job while the arm holds an object

    # ------------------------------------------------------------ sensing
    def true_pose(self):
        d = self.w.d
        R = d.xmat[self.ch].reshape(3, 3)
        x, y, z = d.xpos[self.ch]
        yaw = float(np.arctan2(R[1, 0], R[0, 0]))
        return float(x), float(y), yaw, R

    def pose(self):
        """Localisation estimate (x, y, yaw): truth + small noise, like a fused odometry/IMU estimate."""
        x, y, yaw, _ = self.true_pose()
        n = self.rng.normal(0, self.pose_noise, 3)
        return x + n[0], y + n[1], float(wrap(yaw + 2 * n[2]))

    def imu_tilt(self):
        """(pitch along +x, roll along +y) of the chassis in rad: what the IMU reads."""
        R = self.true_pose()[3]
        z = R.T @ np.array([0, 0, 1.0])
        return float(np.arctan2(-z[0], z[2])), float(np.arctan2(z[1], z[2]))

    # ------------------------------------------------------------ actuation
    def brake(self, on: bool):
        self.stop()
        self.w.brake(self.rid, on)
        self.braked = on

    def stop(self):
        d = self.w.d
        if self.holonomic:
            d.ctrl[self.kin.acts] = 0.0
        else:
            d.ctrl[self.kin.drive] = 0.0               # wheels stop; steering keeps its angle

    def drive(self, vx, vy, wz, seconds: float, monitor=None) -> dict:
        """Hold a body twist (chassis frame) for `seconds`; returns what actually happened, in the
        chassis frame at the start: displacement, heading change, peak tilt.
        monitor(t) -> None | reason: called every 10 physics steps (the safety shield); a reason
        stops the wheels at once, lets the rover settle 0.1 s and brakes ("shield", "elapsed")."""
        if self.braked:
            self.w.brake(self.rid, False)
            self.braked = False
        x0, y0, yaw0, _ = self.true_pose()
        d = self.w.d
        w_prev = d.ctrl[self.kin.acts].copy()
        w_new = self.kin.command(d, vx, vy, wz)
        n = max(1, int(seconds / self.w.m.opt.timestep))
        n_ramp = int(self.RAMP_S / self.w.m.opt.timestep)    # acceleration limit: blend the commands
        tilt, why, dt = 0.0, None, self.w.m.opt.timestep
        elapsed = n * dt
        for k in range(0, n, 10):
            s = min(1.0, (k + 10) / max(n_ramp, 1))
            d.ctrl[self.kin.acts] = w_prev + s * (w_new - w_prev)
            self.w.step(min(10, n - k))
            R = self.true_pose()[3]
            tilt = max(tilt, float(np.arccos(np.clip(R[2, 2], -1, 1))))
            if monitor is not None:
                why = monitor((k + 10) * dt)
                if why:
                    elapsed = (k + 10) * dt
                    self.stop()
                    self.w.step(int(0.1 / dt))
                    self.brake(True)
                    break
        x1, y1, yaw1, _ = self.true_pose()
        c, s = np.cos(yaw0), np.sin(yaw0)
        dx, dy = x1 - x0, y1 - y0
        return {"dx": c * dx + s * dy, "dy": -s * dx + c * dy, "dyaw": float(wrap(yaw1 - yaw0)), "tilt_max": tilt,
                "x": x1, "y": y1, "yaw": yaw1, "shield": why, "elapsed": elapsed}

    # ------------------------------------------------------------ steering law
    def twist_toward(self, heading, speed, yaw, d_omega=0.0, k_yaw=2.0):
        """Chassis twist that moves along world `heading` at `speed`. Holonomic: just go that way.
        Steered: turn toward it (in place when it is far off), and drive an arc."""
        wmax = 0.8 * self.kin.w_max
        if self.holonomic:
            c, s = np.cos(yaw), np.sin(yaw)
            vx, vy = speed * (c * np.cos(heading) + s * np.sin(heading)), speed * (-s * np.cos(heading) + c * np.sin(heading))
            return vx, vy, float(np.clip(d_omega, -wmax, wmax))
        err = float(wrap(heading - yaw))
        if speed < 0:                                       # reversing: the heading is behind us
            err = float(wrap(heading + np.pi - yaw))
        wz = float(np.clip(k_yaw * err + d_omega, -wmax, wmax))
        if abs(err) > self.TURN_IN_PLACE:
            return 0.0, 0.0, wz
        return float(speed * np.cos(err)), 0.0, wz

    # ------------------------------------------------------------ nominal (scripted) control
    def nominal_twist(self, target_xy, target_yaw=None, speed=None, pose=None):
        """The reference command toward a point (the 'recording' a learned drive skill corrects)."""
        x, y, yaw = pose if pose is not None else self.pose()
        vmax = 0.8 * self.kin.v_max if speed is None else speed
        e = np.array(target_xy, float) - [x, y]
        dist = float(np.linalg.norm(e))
        v = min(vmax, 1.5 * dist)
        heading = float(np.arctan2(e[1], e[0])) if dist > 1e-6 else yaw
        if self.holonomic:
            vx, vy, _ = self.twist_toward(heading, v, yaw)
            wz = 0.0 if target_yaw is None else float(np.clip(2.0 * wrap(target_yaw - yaw), -0.8 * self.kin.w_max,
                                                                0.8 * self.kin.w_max))
            return vx, vy, wz, dist
        vx, vy, wz = self.twist_toward(heading, v, yaw)
        return vx, vy, wz, dist

    def turn_to(self, yaw_target, tol=0.02, max_s=None, seg=0.1, monitor=None):
        """Point turn to yaw_target. monitor: the shield's watcher (Rover.drive); a stop -> False."""
        max_s = max_s or 8.0 / self.kin.w_max          # time budgets scale with this design's speed
        t = 0.0
        while t < max_s:
            err = float(wrap(yaw_target - self.pose()[2]))
            if abs(err) < tol:
                return True
            wz = float(np.clip(2.0 * err, -0.8 * self.kin.w_max, 0.8 * self.kin.w_max))
            if abs(wz) < 0.08:
                wz = 0.08 * np.sign(err)
            if self.drive(0.0, 0.0, wz, seg, monitor=monitor).get("shield"):
                return False
            t += seg
        return False

    def goto(self, x, y, yaw=None, tol=0.006, yaw_tol=0.02, max_s=None, seg=0.1, speed=None, line_up=None):
        """Scripted closed-loop move (docking and short hops). Returns True when it arrived.
        line_up: how far behind the pose a non-holonomic base lines up (default LINE_UP)."""
        max_s = max_s or 6.0 / self.kin.v_max + 4 * np.pi / self.kin.w_max   # driving and turning time: a slow
                                                       # or slow-turning design (rocker-bogie scrubs) gets longer
        if self.holonomic:
            t = 0.0
            while t < max_s:
                vx, vy, wz, dist = self.nominal_twist((x, y), yaw, speed)
                dyaw = 0.0 if yaw is None else abs(float(wrap(yaw - self.pose()[2])))
                if dist < tol and dyaw < yaw_tol:
                    break
                self.drive(vx, vy, wz, seg)
                t += seg
            self.brake(True)
            return t < max_s
        return self._dock_steered(x, y, yaw, max(tol, 0.012), yaw_tol, max_s, seg, speed,
                                  self.LINE_UP if line_up is None else line_up)

    LINE_UP = 0.40       # m: a steered / skid-steered base lines up this far behind the dock pose
    DOCK_TRIES = 3       # creep-ins: a miss backs up straight and creeps in again

    def _dock_steered(self, x, y, yaw, tol, yaw_tol, max_s, seg, speed, line_up):
        """A non-holonomic rover cannot move sideways: line up behind the pose on its heading, turn to
        the heading, creep straight in with cross-track correction, square up. A skid-steered rover's
        pivot turns shift it sideways a little, so if it ends off the pose it backs up straight along
        the heading and creeps in again (the cross-track correction needs distance to work)."""
        vmax = 0.5 * self.kin.v_max if speed is None else speed
        if yaw is None:
            ok = self._drive_to_point(x, y, tol, max_s, seg, vmax)
            self.brake(True)
            return ok
        u = np.array([np.cos(yaw), np.sin(yaw)])
        n = np.array([-u[1], u[0]])

        def frame():
            px, py, pyaw = self.pose()
            rel = np.array([px - x, py - y])
            return float(rel @ u), float(rel @ n), pyaw

        along, cross, pyaw = frame()
        if not (-line_up - 0.05 < along < 0.02 and abs(cross) < 0.03 and abs(wrap(pyaw - yaw)) < 0.25):
            pre = np.array([x, y]) - line_up * u
            self._drive_to_point(pre[0], pre[1], 0.02, max_s / 2, seg, vmax)
        t = 0.0
        max_s = max_s + line_up / max(vmax, 1e-3)               # a long line-up needs the creep time
        for attempt in range(self.DOCK_TRIES):
            self.turn_to(yaw, tol=yaw_tol)
            while t < max_s:                                    # creep in along the heading
                along, cross, pyaw = frame()
                if along > -tol:
                    break
                err = float(wrap(yaw - pyaw)) - float(np.clip(4.0 * cross, -0.35, 0.35))
                v = min(vmax, 1.2 * -along + 0.01)
                self.drive(v, 0.0, float(np.clip(2.5 * err, -0.5 * self.kin.w_max, 0.5 * self.kin.w_max)), seg)
                t += seg
            self.turn_to(yaw, tol=yaw_tol)
            along, cross, _ = frame()
            if np.hypot(along, cross) < 3 * tol or t >= max_s or attempt == self.DOCK_TRIES - 1:
                break
            a0 = along                                          # missed: back up straight and try again
            while a0 - frame()[0] < min(0.25 + 4 * abs(cross), 0.6) and t < max_s:
                self.drive(-0.5 * vmax, 0.0, 0.0, seg)
                t += seg
        self.brake(True)
        px, py, _ = self.pose()
        return bool(np.hypot(px - x, py - y) < 3 * tol)

    def back_out(self, x, y, yaw, dist, seg=0.1, speed=None):
        """Reverse straight along the line through (x, y) with heading yaw (the bay's park pose)
        until `dist` behind it, holding the heading and the line (a skid-steer drifts sideways)."""
        vmax = 0.5 * self.kin.v_max if speed is None else speed
        u = np.array([np.cos(yaw), np.sin(yaw)])
        n = np.array([-u[1], u[0]])
        t, max_s = 0.0, 3.0 * dist / vmax + 5.0
        while t < max_s:
            px, py, pyaw = self.pose()
            rel = np.array([px - x, py - y])
            along, cross = float(rel @ u), float(rel @ n)
            if along < -dist:
                break
            err = float(wrap(yaw - pyaw)) + float(np.clip(4.0 * cross, -0.35, 0.35))   # reversing: the sign flips
            v = min(vmax, 1.5 * (dist + along) + 0.02)
            self.drive(-v, 0.0, float(np.clip(2.5 * err, -0.5 * self.kin.w_max, 0.5 * self.kin.w_max)), seg)
            t += seg
        self.brake(True)
        return t < max_s

    def _drive_to_point(self, x, y, tol, max_s, seg, vmax):
        t = 0.0
        while t < max_s:
            vx, vy, wz, dist = self.nominal_twist((x, y), None, vmax)
            if dist < tol:
                return True
            self.drive(vx, vy, wz, seg)
            t += seg
        return False

    def follow(self, waypoints, yaw=None, speed=None, **kw):
        """Scripted: pass through waypoints (loosely), dock exactly at the last one."""
        for wx, wy in waypoints[:-1]:
            self.goto(wx, wy, None, tol=0.05, speed=speed, max_s=kw.get("max_s"))
        return self.goto(*waypoints[-1], yaw, speed=speed, **kw)

    def teleport(self, x, y, yaw):
        """Simulation reset only (check scripts / episode resets), never used by a skill."""
        d = self.w.d
        was = self.braked
        self.w.brake(self.rid, False)
        gz = max(self.w.ground_z(x + dx, y + dy) for dx in (-0.25, 0, 0.25) for dy in (-0.2, 0, 0.2))   # on the terrain
        d.qpos[self.qadr:self.qadr + 7] = [x, y, gz + W.probe(self.design).chassis_z + 0.01, np.cos(yaw / 2), 0, 0, np.sin(yaw / 2)]
        d.qvel[self.vadr:self.vadr + 6] = 0
        mujoco.mj_forward(self.w.m, d)
        self.w.step(250)
        if was:
            self.brake(True)
