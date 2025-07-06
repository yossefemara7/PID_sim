import streamlit as st
import numpy as np
import pandas as pd
import time

st.set_page_config(layout="centered")
st.title("Real-Time PID Simulation SRMP UAV Course")

# --- Persistent message after rerun ---
if "auto_tune_message" in st.session_state:
    st.success(st.session_state["auto_tune_message"])
    del st.session_state["auto_tune_message"]

# --- Fixed Time Step ---
fps = 30
dt = 1.0 / fps

# --- PID Sliders ---
Kp = st.slider("K (Proportional)", 0.0, 5.0,
               st.session_state.get("Kp_new", 2.4), 0.1, key="Kp_slider")
Ki = st.slider("Ki (Integral)", 0.0, 1.5,
               st.session_state.get("Ki_new", 0.61), 0.01, key="Ki_slider")
Kd = st.slider("Kd (Derivative)", 0.0, 1.0,
               st.session_state.get("Kd_new", 0.10), 0.01, key="Kd_slider")

# --- System Parameters ---
mass = st.slider("Mass (kg)", 0.1, 5.0, 1.0, 0.1)
damping = st.slider("Damping", 0.0, 1.0, 0.2, 0.05)
noise_toggle = st.checkbox("Add Sensor Noise", True)
show_velocity = st.checkbox("Show Velocity on Chart", False)
duration = st.slider("Simulation Duration (s)", 1, 30, 10)

# --- Buttons ---
col1, col2, col3, col4 = st.columns(4)
start = col1.button("▶️ Start")
disturb = col2.button("💥 Inject Disturbance")
reset = col3.button("🔄 Reset")
auto_tune = col4.button("🎛️ Brute-Force Auto Tune")

# --- Session State Init ---
if "running" not in st.session_state:
    st.session_state.running = False
if "t" not in st.session_state or reset:
    st.session_state.t = 0.0
    st.session_state.pv = 0.0
    st.session_state.velocity = 0.0
    st.session_state.integral = 0.0
    st.session_state.prev_error = 0.0
    st.session_state.data = pd.DataFrame(columns=["Time", "Setpoint", "PV", "Output", "Velocity"])
    st.session_state.running = False
if "inject_disturbance" not in st.session_state:
    st.session_state.inject_disturbance = False

# --- Chart ---
display_cols = ["Setpoint", "PV", "Output"]
if show_velocity:
    display_cols.append("Velocity")
chart = st.line_chart(st.session_state.data.set_index("Time")[display_cols])

# --- PID Step ---
def pid_step():
    setpoint = 1.0
    error = setpoint - st.session_state.pv
    st.session_state.integral += error * dt
    derivative = (error - st.session_state.prev_error) / dt
    output = Kp * error + Ki * st.session_state.integral + Kd * derivative
    output = np.clip(output, -3, 3)

    if noise_toggle:
        output += np.random.normal(0, 0.01)

    if st.session_state.inject_disturbance:
        st.session_state.velocity -= 1.0
        st.session_state.inject_disturbance = False

    acceleration = (output - damping * st.session_state.velocity) / mass
    st.session_state.velocity += acceleration * dt
    st.session_state.pv += st.session_state.velocity * dt
    st.session_state.pv = np.clip(st.session_state.pv, -3, 3)

    st.session_state.prev_error = error
    st.session_state.t += dt

    row = pd.DataFrame({
        "Time": [st.session_state.t],
        "Setpoint": [setpoint],
        "PV": [st.session_state.pv],
        "Output": [output],
        "Velocity": [st.session_state.velocity]
    })
    st.session_state.data = pd.concat([st.session_state.data, row], ignore_index=True)
    chart.add_rows(row.set_index("Time")[display_cols])

# --- Button Logic ---
if start:
    st.session_state.running = True
if disturb:
    st.session_state.inject_disturbance = True
if reset:
    st.session_state.running = False

# --- Brute-Force Auto-Tune ---
if auto_tune:
    st.session_state.running = False
    st.success("Running Brute-Force Auto-Tune...")

    best_score = float("inf")
    best_params = None

    kp_range = np.arange(0.1, 3.1, 0.3)
    ki_range = np.arange(0.0, 1.1, 0.2)
    kd_range = np.arange(0.0, 1.1, 0.2)

    best_result_text = st.empty()
    progress = st.progress(0.0)
    total = len(kp_range) * len(ki_range) * len(kd_range)
    tested = 0

    for kp_test in kp_range:
        for ki_test in ki_range:
            for kd_test in kd_range:
                pv = 0.0
                velocity = 0.0
                integral = 0.0
                prev_error = 0.0
                t = 0.0
                errors = []
                max_pv = -float("inf")
                setpoint = 1.0

                for _ in range(int(10 / dt)):
                    error = setpoint - pv
                    integral += error * dt
                    derivative = (error - prev_error) / dt
                    output = kp_test * error + ki_test * integral + kd_test * derivative
                    output = np.clip(output, -3, 3)
                    acceleration = (output - damping * velocity) / mass
                    velocity += acceleration * dt
                    pv += velocity * dt
                    pv = np.clip(pv, -3, 3)
                    prev_error = error
                    t += dt
                    max_pv = max(max_pv, pv)
                    errors.append(abs(error))

                    if t > 3:
                        recent = errors[-int(1 / dt):]
                        if all(e < 0.02 for e in recent):
                            break

                overshoot = max_pv - setpoint
                total_error = sum(errors)
                score = t + overshoot + 0.5 * total_error

                if score < best_score:
                    best_score = score
                    best_params = (kp_test, ki_test, kd_test)
                    best_result_text.info(f"Best So Far → Kp: {kp_test:.2f}, Ki: {ki_test:.2f}, Kd: {kd_test:.2f}, Score: {score:.2f}")

                tested += 1
                progress.progress(tested / total)

    progress.empty()

    if best_params:
        Kp_new, Ki_new, Kd_new = best_params
        st.session_state["Kp_new"] = Kp_new
        st.session_state["Ki_new"] = Ki_new
        st.session_state["Kd_new"] = Kd_new

        st.session_state.t = 0.0
        st.session_state.pv = 0.0
        st.session_state.velocity = 0.0
        st.session_state.integral = 0.0
        st.session_state.prev_error = 0.0
        st.session_state.data = pd.DataFrame(columns=["Time", "Setpoint", "PV", "Output", "Velocity"])

        st.session_state["auto_tune_message"] = f"🎯 Best PID Found → Kp: {Kp_new:.2f}, Ki: {Ki_new:.2f}, Kd: {Kd_new:.2f}"
        st.rerun()
    else:
        st.error("Brute-force tuning failed. Try expanding the range or simulation time.")

# --- Run Main Simulation ---
if st.session_state.running:
    for _ in range(int(duration * fps)):
        pid_step()
        time.sleep(dt)
