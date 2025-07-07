import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import io
from PIL import Image

st.set_page_config(layout="centered")
st.title("Real-Time PID Simulation SRMP UAV Course")

# --- Persistent Message After Rerun ---
if "auto_tune_message" in st.session_state:
    st.success(st.session_state["auto_tune_message"])
    del st.session_state["auto_tune_message"]

# --- Simulation Parameters ---
fps = 120 * 4
dt = 10.0 / fps
draw_interval = 10

# --- PID Parameters (User-Controlled) ---
Kp = st.slider("K (Proportional)", 0.0, 5.0, st.session_state.get("Kp_new", 2.4), 0.1, key="Kp_slider")
Ki = st.slider("Ki (Integral)", 0.0, 1.5, st.session_state.get("Ki_new", 0.61), 0.01, key="Ki_slider")
Kd = st.slider("Kd (Derivative)", 0.0, 1.0, st.session_state.get("Kd_new", 0.10), 0.01, key="Kd_slider")

# --- System Parameters ---
mass = 0.75
damping = st.slider("Damping", 0.0, 1.0, 0.2, 0.05)
sensor_bias = st.slider("Sensor Bias", -0.2, 0.2, 0.05, 0.01)
external_force = st.slider("External Force", -0.5, 0.5, 0.15, 0.01)
disturbance_strength = st.slider("Disturbance Strength", 0.1, 5.0, 1.0, 0.1)
noise_toggle = st.checkbox("Add Sensor Noise", True)
show_velocity = st.checkbox("Show Velocity on Chart", False)
duration = st.slider("Simulation Duration (s)", 1, 30, 10)

# --- Control Buttons ---
col1, col2, col3, col4 = st.columns(4)
start = col1.button("▶️ Start")
disturb = col2.button("💥 Inject Disturbance")
reset = col3.button("🔄 Reset")
auto_tune = col4.button("🎛️ Brute-Force Auto Tune")

# --- Session State Initialization ---
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

# --- Brute-Force Auto Tune ---
if auto_tune:
    st.session_state.running = False
    st.success("Running Brute-Force Auto-Tune...")

    best_score = float("inf")
    best_params = None

    kp_range = np.arange(0.0, 4.1, 0.3)
    ki_range = np.arange(0.0, 1.1, 0.05)
    kd_range = np.arange(0.0, 4.1, 0.3)

    best_result_text = st.empty()
    progress = st.progress(0.0)
    total = len(kp_range) * len(ki_range) * len(kd_range)
    tested = 0

    tune_dt = 0.02
    tune_steps = int(6 / tune_dt)

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
                setpoint = [1.0, -1.0, 0.5, -0.5][int(st.session_state.t // 1) % 4]

                for _ in range(tune_steps):
                    measured_pv = pv + sensor_bias
                    error = setpoint - measured_pv
                    integral += error * tune_dt
                    derivative = (error - prev_error) / tune_dt
                    output = kp_test * error + ki_test * integral + kd_test * derivative
                    output = np.clip(output, -3, 3)
                    acceleration = (output - damping * velocity + external_force) / mass
                    velocity += acceleration * tune_dt
                    pv += velocity * tune_dt
                    pv = np.clip(pv, -3, 3)
                    prev_error = error
                    t += tune_dt
                    max_pv = max(max_pv, pv)
                    errors.append(abs(error))

                    if t > 4:
                        recent = errors[-int(1 / tune_dt):]
                        if all(e < 0.02 for e in recent):
                            break

                overshoot = max_pv - setpoint
                total_error = sum(errors)
                recent_error = np.mean(errors[-int(1.0 / tune_dt):])
                score = t + 0.5 * overshoot + 0.5 * total_error + 25.0 * recent_error

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

# --- Line Chart ---
display_cols = ["Setpoint", "PV", "Output"]
if show_velocity:
    display_cols.append("Velocity")
chart = st.line_chart(st.session_state.data.set_index("Time")[display_cols])

# --- Physical Representation ---
st.subheader("Physical Representation (Position on Track)")
track_placeholder = st.empty()
def get_current_setpoint():
    return [1.0, -1.0, 0.5, -0.5][int(st.session_state.t // 10) % 4]

def draw_physical_system(pv_value, setpoint_val):
    fig, ax = plt.subplots(figsize=(6, 1.2))
    ax.set_xlim(-3, 3)
    ax.set_ylim(-0.5, 0.5)
    ax.axis('off')
    ax.hlines(0, -3, 3, colors='gray', linewidth=4)
    ax.axvline(setpoint_val, color='red', linestyle='--', linewidth=2)
    ax.text(setpoint_val, 0.3, '🎯 Setpoint', color='red', ha='center', fontsize=10)
    try:
        drone_img = plt.imread("drone.png")
        img_extent = [pv_value - 0.4, pv_value + 0.4, -0.4, 0.4]
        ax.imshow(drone_img, extent=img_extent, aspect='auto', zorder=5)
    except:
        ax.plot(pv_value, 0, 'o', markersize=20, color='blue')
        st.warning("Drone image not found, showing dot instead.")
    buf = io.BytesIO()
    plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0.1)
    buf.seek(0)
    img = Image.open(buf)
    track_placeholder.image(img)
    plt.close(fig)


# --- PID Step Simulation ---
step_counter = 0
def pid_step():
    global step_counter
    setpoint = get_current_setpoint()
    measured_pv = st.session_state.pv + sensor_bias
    error = setpoint - measured_pv
    st.session_state.integral += error * dt
    derivative = (error - st.session_state.prev_error) / dt
    output = Kp * error + Ki * st.session_state.integral + Kd * derivative
    output = np.clip(output, -3, 3)
    if noise_toggle:
        output += np.random.normal(0, 0.05)
    if st.session_state.inject_disturbance:
        st.session_state.velocity -= disturbance_strength
        st.session_state.inject_disturbance = False
    acceleration = (output - damping * st.session_state.velocity + external_force) / mass
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
    st.session_state.data = st.session_state.data.tail(1000).reset_index(drop=True)

    step_counter += 1
    if step_counter % draw_interval == 0:
        chart.add_rows(row.set_index("Time")[display_cols])
        draw_physical_system(st.session_state.pv, setpoint)


# --- Button Logic ---
if start:
    st.session_state.running = True
if disturb:
    st.session_state.inject_disturbance = True
if reset:
    st.session_state.running = False

# --- Main Loop ---
if st.session_state.running:
    for _ in range(int(duration * fps)):
        pid_step()
