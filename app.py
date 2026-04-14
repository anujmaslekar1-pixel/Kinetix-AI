import os
import hashlib
import time
import streamlit as st
from google import genai
from PIL import Image
from dotenv import load_dotenv
from sqlalchemy import create_engine, Column, Integer, String, Text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from fpdf import FPDF
import cv2
import mediapipe as mp
import numpy as np
import tempfile
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import moviepy as me
import io
import base64
import plotly.graph_objects as go
import urllib.request

if "data_complete" not in st.session_state:
    st.session_state.data_complete = False

# --- 1. DATABASE SETUP ---
engine = create_engine('sqlite:///kinetix_final.db', connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    username = Column(String, unique=True, index=True)
    password = Column(String)
    
class ExerciseAudit(Base):
    __tablename__ = "exercise_audits"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    exercise_name = Column(String)
    feedback_text = Column(Text)


class FitnessPlan(Base):
    __tablename__ = "fitness_plans"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer)
    name = Column(String)
    goal = Column(String)
    plan_text = Column(Text)

Base.metadata.create_all(bind=engine)

# --- 2. THE AI ENGINE & LOGIC FUNCTIONS ---

def hash_password(password):
    return hashlib.sha256(str.encode(password)).hexdigest()

def check_login(username, password):
    db = SessionLocal()
    user = db.query(User).filter(User.username == username, User.password == hash_password(password)).first()
    db.close()
    return user

def calculate_macros(weight, height, age, goal):
    bmr = (10 * weight) + (6.25 * height) - (5 * age) + 5
    maint = int(bmr * 1.55)
    if goal == "Muscle Gain": target = maint + 400
    elif goal == "Fat Loss": target = maint - 500
    else: target = maint
    return maint, target

def generate_fitness_plan(u_name, u_goal, u_cons, u_exp, u_focus, stats, demo_mode, client):
    """Elite Coach AI Logic: Focused on Tiered Training, Focus Areas, & Injury Bio-mechanics."""
    if demo_mode:
        return f"### [DEMO MODE] {u_goal} Strategy for {u_name}\n- **Injury Focus:** Avoiding stress on {u_cons}."

    # 1. Experience-Specific Training Philosophy (Same as your current logic)
    exp_matrix = {
        "Beginner": {
            "split": "3-Day Full Body Split (Mon/Wed/Fri)",
            "volume": "8-10 weekly sets per muscle group",
            "progression": "Linear (Add small weight every session)",
            "frequency": "3 Days Work, 2 Active Recovery, 2 Full Rest"
        },
        "Intermediate": {
            "split": "4-Day Upper/Lower Split",
            "volume": "12-16 weekly sets per muscle group",
            "progression": "Double Progression (Reps then Weight)",
            "frequency": "4 Days Work, 2 Active Recovery, 1 Full Rest"
        },
        "Advanced": {
            "split": "6-Day PPL or Specialized Split",
            "volume": "18-22 weekly sets per muscle group",
            "progression": "Periodization & RPE-based training",
            "frequency": "6 Days Work, 1 Active Recovery"
        }
    }
    
    tier = exp_matrix.get(u_exp, exp_matrix["Beginner"])

    # 2. HANDLE INJURY LOGIC (The Fix)
    if not u_cons or u_cons.lower() in ["none", "n/a", "no", "nothing"]:
        injury_instruction = f"The user has NO injuries. Provide a high-performance, unrestricted elite program focusing on {u_goal} and {u_focus}."
        safety_section_name = "Performance Optimization"
    else:
        injury_instruction = f"""
        INJURY MANAGEMENT PROTOCOL:
        - Analyze the biomechanics of '{u_cons}'. 
        - Replace high-impact movements with low-impact or isometric alternatives.
        - Add a 'Physio Note' for exercises involving the kinetic chain of the injured area.
        """
        safety_section_name = "Safety & Rehab (Injury-Specific)"

    MODEL_PRIORITY = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-1.5-flash"]
    
    prompt_text = f"""
    ROLE: DPT (Doctor of Physical Therapy) and Elite Strength Coach.
    USER: {u_name} | GOAL: {u_goal} | EXPERIENCE: {u_exp} | TARGET FOCUS: {u_focus}

    {injury_instruction}

    TRAINING STRUCTURE ({u_exp.upper()}):
    - Weekly Layout: {tier['split']}
    - Target Volume: {tier['volume']}
    - Frequency: {tier['frequency']}
    - Progression: {tier['progression']}
    - Priority: Incorporate specialized variations for the '{u_focus}'.

    OUTPUT REQUIREMENTS:
    1. A 7-Day Table: [Day, Exercise, Sets x Reps, Rest, Coach's Note].
    2. A 'Weekly Volume Summary' showing total sets assigned per muscle group.
    3. A '{safety_section_name}' section explaining the movement selection.
    4. A 'Future Perspective' section advising on reassessment after 8-12 weeks.
    5. NO NUTRITION ADVICE.
    6. Format in professional Markdown.
    """

    for model_id in MODEL_PRIORITY:
        try:
            response = client.models.generate_content(model=model_id, contents=prompt_text)
            return response.text
        except:
            continue
    return "Engine error. Please try again."


def export_pdf(plan_text, user_name):
    pdf = FPDF()
    pdf.add_page()

    # Helper function to prevent Latin-1 encoding crashes
    def safe_text(s):
        if not s: return ""
        # Replace common fancy characters manually for a better look
        s = s.replace('\u2013', '-').replace('\u2014', '-').replace('\u2019', "'").replace('\u2018', "'")
        # Final safety check to replace anything else with '?'
        return s.encode('latin-1', 'replace').decode('latin-1')
    
    # 1. Header Styling
    pdf.set_fill_color(76, 175, 80) # Kinetix Green
    pdf.set_text_color(255, 255, 255)
    pdf.set_font("Courier", 'B', 16)
    # Applied safe_text to header
    pdf.cell(0, 12, txt=safe_text(f"KINETIX AI: {user_name.upper()} PLAN"), ln=True, align='C', fill=True)
    
    pdf.set_text_color(0, 0, 0)
    pdf.ln(5)
    
    lines = plan_text.split('\n')
    for line in lines:
        line = line.strip()
        if not line: continue
        
        # 2. DETECT AND RENDER TABLES
        if '|' in line and '---' not in line:
            cols = [c.strip().replace('**', '') for c in line.split('|') if c.strip()]
            
            if len(cols) >= 2:
                w_ex = 80  
                w_sets = 40 
                w_rest = 30 
                
                pdf.set_font("Courier", 'B', 9)
                
                # FIX: Applied safe_text to every table cell
                pdf.cell(w_ex, 8, txt=safe_text(cols[0][:40]), border=1)
                pdf.cell(w_sets, 8, txt=safe_text(cols[1] if len(cols) > 1 else ""), border=1)
                if len(cols) > 2:
                    pdf.cell(w_rest, 8, txt=safe_text(cols[2]), border=1)
                
                pdf.ln()
                
                # FIX: Applied safe_text to the Note section
                if len(cols) > 3:
                    pdf.set_font("Courier", 'I', 8)
                    pdf.set_text_color(100, 100, 100)
                    pdf.multi_cell(0, 5, txt=safe_text(f"   Note: {cols[3]}"))
                    pdf.set_text_color(0, 0, 0)
                    pdf.ln(2)
        
        # 3. REGULAR TEXT (Headers and Paragraphs)
        else:
            pdf.set_font("Courier", 'B' if line.startswith('#') else '', 10)
            clean_line = line.replace("#", "").replace("**", "").strip()
            # Applied safe_text to regular text
            pdf.multi_cell(0, 6, txt=safe_text(clean_line))
            pdf.ln(1)

    return pdf.output(dest='S').encode('latin-1')

def get_macros_from_text(text_input, client):
    """Parses natural language into JSON macros for plotting"""
    MODEL_PRIORITY = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-1.5-flash"]
    
    prompt = f"""
    Act as a nutrition database. Extract macros from this text: {text_input}
    Return ONLY a JSON object. If multiple items, sum them.
    Format: {{"Protein": 30, "Carbs": 50, "Fats": 10, "Calories": 410}}
    """
    
    for model_id in MODEL_PRIORITY:
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            import json
            clean_json = response.text.strip().replace("```json", "").replace("```", "")
            return json.loads(clean_json)
        except:
            continue
    return None

def generate_diet_only_plan(u_name, diet_type, goal, requests, stats, client):
    MODEL_PRIORITY = ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-1.5-flash"]
    target_kcal = stats.get('target', 'Not specified')
    
    prompt = f"""
    ROLE: Clinical Nutritionist and Sports Dietitian.
    USER PROFILE: {u_name} | {diet_type} diet | Nutrition Goal: {goal}.
    DAILY CALORIE TARGET: {target_kcal} kcal
    SPECIAL CONSTRAINTS/ALLERGIES: {requests}
    TASK: Provide a high-protein 1-day sample meal plan that precisely hits the target of {target_kcal} kcal.
    STRICT RULES:
    1. Every meal must contribute toward the {target_kcal} kcal limit.
    2. For EVERY MEAL, provide a breakdown: [Calories, Protein(g), Carbs(g), Fats(g)].
    3. DIET RULES:
       - If 'Mixed': Include a variety of sources (chicken, fish, eggs, dairy, plants).
       - If 'Veg': NO meat/fish/eggs. Use paneer, legumes, dairy.
       - If 'Eggitarian': NO meat/fish. Eggs and dairy are allowed.
    4. Respect all allergies: {requests}.
    5. At the end, provide a 'DAILY TOTAL' summary.
    6. Progression: This plan should be followed for 4-6 weeks.
    7. Format using professional Markdown with clear tables for the meals.
    """
    
    for model_id in MODEL_PRIORITY:
        try:
            response = client.models.generate_content(model=model_id, contents=prompt)
            return response.text
        except:
            continue
    return "Error: AI Nutritionist is currently offline."
def calculate_angle(a, b, c):
    a, b, c = np.array(a), np.array(b), np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians*180.0/np.pi)
    return 360-angle if angle > 180.0 else angle


def draw_skeleton(image, landmarks):
    CONNECTIONS = [
        (11, 13), (13, 15), (12, 14), (14, 16), # Arms
        (11, 23), (23, 25), (25, 27), (12, 24), # Torso/Legs
        (24, 26), (26, 28), (11, 12), (23, 24)  # Widths
    ]
    h, w, _ = image.shape
    for start_idx, end_idx in CONNECTIONS:
        try:
            start = (int(landmarks[start_idx].x * w), int(landmarks[start_idx].y * h))
            end = (int(landmarks[end_idx].x * w), int(landmarks[end_idx].y * h))
            cv2.line(image, start, end, (0, 255, 0), 2)
            cv2.circle(image, start, 4, (255, 255, 255), -1)
        except: continue


def process_video_locally(video_path, exercise_type):
    model_path = os.path.join(os.getcwd(), 'pose_landmarker.task')
    # Download the model if it's missing
    if not os.path.exists(model_path):
        try:
            url = "https://storage.googleapis.com/mediapipe-models/pose_landmarker/pose_landmarker_heavy/float16/1/pose_landmarker_heavy.task"
            urllib.request.urlretrieve(url, model_path)
        except Exception as e:
            return f"Error downloading AI model: {e}"

    files_to_clear = ["output_render.mp4", "final_output.mp4", "audit_result.jpg"]
    for f in files_to_clear:
        if os.path.exists(f):
            try:
                os.remove(f)
            except Exception as e:
                print(f"Could not remove old cache file {f}: {e}")


    base_options = python.BaseOptions(model_asset_path='pose_landmarker.task')
    options = vision.PoseLandmarkerOptions(base_options=base_options, running_mode=vision.RunningMode.VIDEO)
    detector = vision.PoseLandmarker.create_from_options(options)
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps <= 0: fps = 30

    # --- 1. VIDEO WRITER INITIALIZATION ---
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    raw_video_path = "output_render.mp4"
    final_video_path = "final_output.mp4"
    fourcc = cv2.VideoWriter_fourcc(*'mp4v') 
    out = cv2.VideoWriter(raw_video_path, fourcc, fps, (w, h))
    
    # --- Metrics Initialization ---
    min_knee_angle, max_hip_angle, min_back_angle = 180, 0, 180
    hip_dropped_below_knee = False
    bar_x_positions, frame_count = [], 0
    audit_frame = None
    best_peak_val = 0 if "Deadlift" in exercise_type or "Press" in exercise_type else 180

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        timestamp_ms = int((frame_count / fps) * 1000)
        results = detector.detect_for_video(mp_image, timestamp_ms)
        frame_count += 1
        
        if results.pose_landmarks:
            lm = results.pose_landmarks[0]
            
            # DRAW SKELETON ON THE LIVE FRAME (For the video)
            draw_skeleton(frame, lm)
            
            shoulder, hip, knee, ankle, wrist = [lm[11].x, lm[11].y], [lm[23].x, lm[23].y], [lm[25].x, lm[25].y], [lm[27].x, lm[27].y], [lm[15].x, lm[15].y]
            ear = [lm[7].x, lm[7].y]
            
            current_main_angle = 180

            if exercise_type == "Squat":
                angle = calculate_angle(hip, knee, ankle)
                current_main_angle = angle
                if angle < min_knee_angle: min_knee_angle = angle
                if hip[1] >= (knee[1] - 0.02): hip_dropped_below_knee = True
            
            elif "Deadlift" in exercise_type:
                h_angle = calculate_angle(shoulder, hip, knee)
                current_main_angle = h_angle
                if h_angle > max_hip_angle: max_hip_angle = h_angle
                spine_line = calculate_angle(ear, shoulder, hip)
                if h_angle > 110 and spine_line < min_back_angle: min_back_angle = spine_line
            
            elif "Bench" in exercise_type:
                elbow = [lm[13].x, lm[13].y]
                angle = calculate_angle(shoulder, elbow, wrist)
                current_main_angle = angle
                if angle < min_knee_angle: min_knee_angle = angle
                if angle > max_hip_angle: max_hip_angle = angle
                
            elif "Press" in exercise_type:
                elbow = [lm[13].x, lm[13].y]
                angle = calculate_angle(shoulder, elbow, wrist)
                current_main_angle = angle
                if angle > max_hip_angle: max_hip_angle = angle
                body_angle = calculate_angle(shoulder, hip, knee)
                if 80 < angle < 165 and body_angle < min_back_angle: min_back_angle = body_angle
            
            bar_x_positions.append(wrist[0])

            # Peak Frame Capture Logic
            update_visual = False
            if exercise_type == "Squat" or "Bench" in exercise_type:
                if current_main_angle <= best_peak_val:
                    best_peak_val = current_main_angle
                    update_visual = True
            else: # Deadlift/Press
                if current_main_angle >= best_peak_val:
                    best_peak_val = current_main_angle
                    update_visual = True

            if update_visual or audit_frame is None:
                audit_frame = frame.copy()

        # WRITE THE FRAME TO THE VIDEO
        out.write(frame)             

    # --- CLEANUP ---
    cap.release()
    out.release()
    detector.close()

    # --- 2. MOVIEPY CONVERSION (The Black Screen Fix) ---
    time.sleep(1) 
    if os.path.exists(final_video_path):
        try:
            os.remove(final_video_path)
        except:
            pass

    try:
        # Using the correct variable names and standard libx264 codec
        with me.VideoFileClip(raw_video_path) as clip:
            clip.write_videofile(final_video_path, codec="libx264", audio=False, preset="ultrafast", logger=None)
    except Exception as e:
        print(f"Video conversion failed: {e}")

    # Save the Audit Image
    if audit_frame is not None:
        cv2.imwrite("audit_result.jpg", audit_frame)

    # --- Summary Logic ---
    bar_dev = (max(bar_x_positions) - min(bar_x_positions)) * 100 if bar_x_positions else 0
    summary = f"### 🛡️ Kinetix AI Exercise Analysis: {exercise_type}\n"
    
    if "Deadlift" in exercise_type:
        if max_hip_angle >= 168:
            summary += "- ✅ **Lockout:** Full hip extension achieved. Rep counts.\n"
        else:
            summary += f"- ❌ **Lockout:** Incomplete extension ({int(max_hip_angle)}°). Hips did not reach verticality.\n"
        
        if min_back_angle > 125 or min_back_angle == 180:
            summary += "- ✅ **Spine Neutrality:** Back remained straight. No significant rounding detected.\n"
        else:
            summary += f"- ❌ **Safety Warning:** Spinal rounding detected ({int(min_back_angle)}°). Keep your chest up and lats engaged.\n"

    elif exercise_type == "Squat":
        if hip_dropped_below_knee: summary += "- ✅ **Depth:** Professional depth reached.\n"
        else: summary += "- ❌ **Depth:** Failing. Quarter rep detected.\n"

    elif "Bench" in exercise_type:
        if min_knee_angle <= 75:
            summary += "- ✅ **Range of Motion:** Full depth achieved. Bar reached the chest.\n"
        else:
            summary += f"- ❌ **Range of Motion:** Partial rep detected ({int(min_knee_angle)}°). Lower the bar further for a valid rep.\n"
        
        if max_hip_angle >= 160:
            summary += "- ✅ **Lockout:** Full tricep extension at the top.\n"
        else:
            summary += "- ⚠️ **Lockout:** Incomplete extension. Ensure you lock your elbows to finish the rep.\n"

        summary += "- ℹ️ **Safety Note:** Keep your feet planted and maintain a slight arch in your lower back.\n" 
    
    elif "Press" in exercise_type:
        if max_hip_angle >= 165:
            summary += "- ✅ **Extension:** Full overhead lockout achieved.\n"
        else:
            summary += f"- ❌ **Extension:** Incomplete lockout ({int(max_hip_angle)}°).\n"

        if min_back_angle > 158 or min_back_angle == 180:
            summary += "- ✅ **Spine Stability:** Body remained stacked and stable.\n"
        else:
            summary += f"- ❌ **Safety Warning:** Excessive lumbar arching ({int(min_back_angle)}°).\n"

    # --- Bar Path Logic ---
    if len(bar_x_positions) > 5:
        try:
            # We use h, w from the initialization above
            aspect_ratio = w / h
        except:
            aspect_ratio = 0.56 
        
        x_array = np.array(bar_x_positions) * aspect_ratio
        path_variance = np.std(x_array) * 100 
        
        if path_variance < 4.0: 
            summary += "- ✅ **Bar Path:** Vertical travel is stable and efficient.\n"
        else:
            if "Bench" in exercise_type:
                summary += f"- ⚠️ **Bar Path:** Minor drift detected ({path_variance:.1f}%).\n"
            elif "Squat" in exercise_type or "Deadlift" in exercise_type:
                summary += f"- ⚠️ **Bar Path:** Horizontal sway detected ({path_variance:.1f}%).\n"
            elif "Press" in exercise_type:
                summary += f"- ⚠️ **Bar Path:** Horizontal drift detected ({path_variance:.1f}%).\n"
    else:
        summary += "- ℹ️ **Bar Path:** Analysis incomplete.\n"
        
    return summary




def local_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
local_css("style.css")



# --- 3. CONFIGURATION & STATE ---
#load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

LOGO_IMAGE = "k2_logo.png"
st.set_page_config(
    page_title="Kinetix AI", 
    page_icon=LOGO_IMAGE, # This changes the browser tab icon
    layout="wide"
)
local_css("style.css")

if 'logged_in' not in st.session_state:
    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None

# --- 4. LOGIN GATEKEEPER ---
if not st.session_state.logged_in:
    col1, login_box, col3 = st.columns([1, 2, 1])
    
    with login_box:
        # 1. CENTERED MAIN LOGO
        sub_col1, sub_col2, sub_col3 = st.columns([1, 1, 1])
        with sub_col2:
            st.image(LOGO_IMAGE, width=100)
            
        # 2. ALIGNED LOCK + GRADIENT TEXT ON ONE LINE
        # We use base64 to ensure the image displays correctly inside the HTML div
        login_icon_base64 = base64.b64encode(open("login_icon.png", "rb").read()).decode()
        
        st.markdown(
    f"""
    <div style="display: flex; align-items: center; justify-content: center; gap: 15px; margin-top: 20px; margin-bottom: 20px;">
        <img src="data:image/png;base64,{login_icon_base64}" width="50">
        <h1 style="background: linear-gradient(90deg, #66BB6A, #A5D6A7, #81C784); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight: 900; margin: 0; line-height: 1; white-space: nowrap;">
            Kinetix Member Access
        </h1>
    </div>
    """, 
    unsafe_allow_html=True
)
        
        # 3. LOGIN UI (Adding the radio button back in)
        mode = st.radio("Choose Action", ["Login", "Register"], horizontal=True)
        
        # ... rest of your login/register logic ...
        
        if mode == "Register":
            # Using unique keys prevents the browser from linking these to the Login fields
            reg_user = st.text_input("Choose Username", key="reg_u", placeholder="Enter a new username")
            reg_pass = st.text_input("Create Password", type='password', key="reg_p", placeholder="Minimum 6 characters")
            reg_confirm = st.text_input("Confirm Password", type='password', key="reg_cp", placeholder="Confirm password")
            
            if st.button("Create Account", use_container_width=True):
                if not reg_user or not reg_pass:
                    st.warning("Please fill in all fields.")
                elif reg_pass != reg_confirm:
                    st.error("Passwords do not match. Please try again.")
                else:
                    db = SessionLocal()
                    exists = db.query(User).filter(User.username == reg_user).first()
                    if exists:
                        st.error("Username already taken. Try another one.")
                    else:
                        new_user = User(username=reg_user, password=hash_password(reg_pass))
                        db.add(new_user)
                        db.commit()
                        st.success("✅ Registration Successful! Please switch to 'Login' mode to enter.")
                    db.close()
        
        else:
            # Login Mode
            login_user = st.text_input("Username", key="login_u", placeholder="Enter your username")
            login_pass = st.text_input("Password", type='password', key="login_p", placeholder="Enter your password")
            
            if st.button("Login", use_container_width=True):
                user_found = check_login(login_user, login_pass)
                if user_found:
                    st.session_state.logged_in = True
                    st.session_state.user_id = user_found.id
                    st.session_state.username = user_found.username
                    st.rerun()
                else:
                    st.error("❌ Invalid credentials. Please check your username and password.")

    st.stop() # Stops the rest of the app from running until logged inpuyyp

logo_data = base64.b64encode(open(LOGO_IMAGE, "rb").read()).decode()
# --- 5. MAIN APP UI ---
with st.sidebar:
    # --- BRANDING: ICON + GRADIENT TEXT ALIGNED ---
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 20px;">
            <img src="data:image/png;base64,{logo_data}" width="65">
            <h1 style="background: linear-gradient(90deg, #66BB6A, #A5D6A7, #81C784); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight: 850; margin: 0; line-height: 1; white-space: nowrap;">Kinetix AI</h1>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    # ... rest of your sidebar (User Info, Demo Mode, etc.) ...
    
    # 2. User Info
    st.write(f"Logged in: **{st.session_state.get('username', 'User')}**")
    
    # 3. App Controls
    demo_mode = st.toggle("Demo Mode", value=False, help="Skips API calls for testing purposes.")
    
    st.divider()

    # 4. Logout Section
    st.markdown('<div class="logout-btn">', unsafe_allow_html=True)
    if st.button("🚪 Logout", use_container_width=True):
        st.session_state.logged_in = False
        st.rerun()
    st.markdown('</div>', unsafe_allow_html=True)   


# --- 6. TABS ---
tab_profile, tab_plan, tab_food, tab_tracker, tab_exercise, tab_history = st.tabs(["👤 Profile & Setup","🏋️ AI FITNESS PLAN", "🥗AI DIET PLAN", "🗓️DAILY CALORIE TRACKER", "🧘 EXERCISE ANALYZER","🕒 HISTORY"])


profile_icon_data = base64.b64encode(open("profile_icon.png", "rb").read()).decode()
with tab_profile:
    # --- VIBRANT ENERGY (YELLOW-ORANGE) GRADIENT HEADER ---
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
            <img src="data:image/png;base64,{profile_icon_data}" width="50">
            <h1 style="background: linear-gradient(90deg, #FDC830, #F37335, #FF5F6D); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight: 800; margin: 0; line-height: 1;">Athlete Profile</h1>
        </div>
        <hr style="margin-top: 2px; margin-bottom: 20px; border: 0; border-top: 2px solid #eee; opacity: 0.4;">
        """,
        unsafe_allow_html=True
    )
    
    

    # 1. Dashboard View (Visible only after registration)
    if st.session_state.get('data_complete', False):
        # Calculations
        h_m = st.session_state.u_height / 100
        bmi = round(st.session_state.u_weight / (h_m**2), 1)
        
        # TDEE estimate (Mifflin-St Jeor Equation)
        tdee = round((10 * st.session_state.u_weight) + (6.25 * st.session_state.u_height) - (5 * st.session_state.u_age) + 5)
        protein_goal = int(st.session_state.u_weight * 2) 

        # BMI Interpretation Logic
        if bmi < 18.5:
            bmi_cat = "Underweight"
            bmi_clr = "#FF4B4B"
        elif 18.5 <= bmi < 25:
            bmi_cat = "Normal weight"
            bmi_clr = "#39ff14"
        elif 25 <= bmi < 30:
            bmi_cat = "Overweight"
            bmi_clr = "#F7971E"
        else:
            bmi_cat = "Obese"
            bmi_clr = "#ef473a"

        st.success(f"✅ You are registered to **Kinetix**, {st.session_state.u_display_name}! The features are unlocked.")
        
        # Metric Row
        m_col1, m_col2, m_col3 = st.columns(3)
        m_col1.metric("Body Mass Index", f"{bmi}")
        m_col2.metric("Daily Calories (Est.)", f"{tdee} kcal")
        m_col3.metric("Protein Target", f"{protein_goal}g")

        # Explicit Text Breakdown
        st.markdown(f"Your BMI is **{bmi}**, which is considered <span style='color:{bmi_clr}; font-weight:bold;'>{bmi_cat}</span>.", unsafe_allow_html=True)

        # Visual Gauge
        fig = go.Figure(go.Indicator(
            mode = "gauge+number",
            value = bmi,
            domain = {'x': [0, 1], 'y': [0, 1]},
            title = {'text': "BMI Visual Status", 'font': {'size': 20}},
            gauge = {
                'axis': {'range': [15, 40], 'tickwidth': 1, 'tickcolor': "white"},
                'bar': {'color': "#00d4ff"},
                'steps': [
                    {'range': [15, 18.5], 'color': "#FF4B4B"},
                    {'range': [18.5, 25], 'color': "#39ff14"},
                    {'range': [25, 30], 'color': "#F7971E"},
                    {'range': [30, 40], 'color': "#ef473a"}],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': bmi}
            }
        ))
        
        fig.update_layout(
            height=350, 
            margin=dict(t=80, b=50, l=30, r=30), 
            paper_bgcolor='rgba(0,0,0,0)', 
            font={'color': "white"}
        )
        
        st.plotly_chart(fig, use_container_width=True)
        st.divider()
    
    # 2. Onboarding Message (Visible only before registration)
    else:
        st.info("Complete your profile to unlock the Diet and Fitness generators.")
    
    # 3. The Input Form
    with st.form("profile_form"):
        name_input = st.text_input("Full Name", value=st.session_state.get('u_display_name', ""))
        
        col1, col2, col3 = st.columns(3)
        with col1:
            weight_input = st.number_input("Weight (kg)", min_value=30.0, value=st.session_state.get('u_weight', 70.0))
        with col2:
            height_input = st.number_input("Height (cm)", min_value=100.0, value=st.session_state.get('u_height', 170.0))
        with col3:
            age_input = st.number_input("Age", min_value=10, value=st.session_state.get('u_age', 25))
            
        if st.form_submit_button("✅ Save Profile & Unlock Features"):
            if name_input:
                st.session_state.u_display_name = name_input
                st.session_state.u_weight = weight_input
                st.session_state.u_height = height_input
                st.session_state.u_age = age_input
                
                # Turn on the logic gate
                st.session_state.data_complete = True
                st.rerun()
            else:
                st.error("Please enter a name.")

plan_icon_data = base64.b64encode(open("plan_icon.png", "rb").read()).decode()
with tab_plan:
    if not st.session_state.get('data_complete', False):
        st.warning("⚠️ Please complete your **Profile & Setup** tab first to unlock the Fitness Architect.")
    else:
        # 1. THE BRANDED HEADER (Aligned with Icon & Gradient)
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                <img src="data:image/png;base64,{plan_icon_data}" width="50">
                <h1 style="background: linear-gradient(90deg, #FDC830, #F37335, #FF5F6D); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight: 800; margin: 0; line-height: 1;">Elite Fitness Program</h1>
            </div>
            <hr style="margin-top: 2px; margin-bottom: 25px; border: 0; border-top: 2px solid #eee; opacity: 0.4;">
            """,
            unsafe_allow_html=True
        )

        # Retrieve variables from Session State
        u_display_name = st.session_state.u_display_name
        u_weight = st.session_state.u_weight
        u_height = st.session_state.u_height
        u_age = st.session_state.u_age
        
        # Row 1: Goal and Experience
        col1, col2 = st.columns(2)
        with col1:
            u_goal_main = st.selectbox("🎯 Primary Goal", ["Muscle Gain", "Fat Loss", "Body Recomp/Maintenance"], key="plan_goal")
        with col2:
            u_exp_main = st.selectbox("📊 Current Level", ["Beginner (0-1 yrs experience)", "Intermediate (0-3 yrs experience)", "Advanced (3+ yrs experience)"], key="plan_exp")
        
        # Row 2: Focus Area
        u_focus = st.text_input("🎯 Target Focus Area (Optional)", 
                                placeholder="e.g., Shoulders, Leg Strength, Back Width", 
                                help="The AI will prioritize this area while keeping the plan balanced.",
                                key="plan_focus")
        
        # Row 3: Injury
        u_cons_main = st.text_area("🩹 Injury Section / Constraints", 
                                   value=st.session_state.get('u_cons'),
                                   placeholder="e.g., Ankle ligament tear, lower back sensitivity, or Leave Empty if None...", 
                                   help="The AI will bypass these areas to prevent further injury.",
                                   key="plan_cons")

        # 2. THE LOCAL GENERATE BUTTON
        if st.button("🔥 Generate Personalized Program", use_container_width=True):
            
            with st.spinner("Elite Coach AI is analyzing your profile..."):
                # Calculate macros locally so 'stats', 'maint', and 'target' exist
                maint, target = calculate_macros(u_weight, u_height, u_age, u_goal_main)
                stats = {'target': target, 'maint': maint}

                # Call the function using u_display_name
                plan_text = generate_fitness_plan(u_display_name, u_goal_main, u_cons_main, u_exp_main, u_focus, stats, demo_mode, client)
                
                if plan_text:
                    # SAVE TO SESSION STATE
                    st.session_state.current_plan_text = plan_text
                    st.session_state.plan_stats = {'maint': maint, 'target': target}
                    
                    # Save to Database
                    try:
                        db = SessionLocal()
                        new_plan = FitnessPlan(
                            user_id=st.session_state.user_id,
                            name=u_display_name,
                            goal=u_goal_main,
                            plan_text=plan_text
                        )
                        db.add(new_plan)
                        db.commit()
                        db.close()
                    except NameError:
                        # Handle cases where database variables might not be initialized
                        pass
                        
                    st.success("Plan Generated!")
                else:
                    st.error("AI Models are busy. Try Demo Mode.")

    # 3. DISPLAY LOGIC (Outside the button click)
    if "current_plan_text" in st.session_state:
        # Use u_display_name from session state for the display and PDF
        u_display_name = st.session_state.u_display_name
        
        st.markdown("---")
        st.markdown(st.session_state.current_plan_text)
        
        # EXPORT SECTION
        st.divider()
        try:
            pdf_bytes = export_pdf(st.session_state.current_plan_text, u_display_name)
            st.download_button(
                label="📥 Download Plan as PDF",
                data=pdf_bytes,
                file_name=f"{u_display_name}_Kinetix_Plan.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="workout_export_btn"
            )
        except Exception as e:
            st.error(f"PDF Error: {e}")


food_icon_data = base64.b64encode(open("food_icon.png", "rb").read()).decode()
with tab_food:
    if not st.session_state.get('data_complete', False):
        st.warning("⚠️ Please complete your **Profile & Setup** tab first to unlock Nutrition AI.")
    else:
        # 1. THE BRANDED HEADER (Aligned Left with Icon)
        st.markdown(
            f"""
            <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
                <img src="data:image/png;base64,{food_icon_data}" width="50">
                <h1 style="background: linear-gradient(90deg, #FDC830, #F37335, #FF5F6D); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight: 800; margin: 0; line-height: 1;">Precision Nutrition AI</h1>
            </div>
            <hr style="margin-top: 2px; margin-bottom: 25px; border: 0; border-top: 2px solid #eee; opacity: 0.4;">
            """,
            unsafe_allow_html=True
        )

        # Retrieve stored variables
        u_weight = st.session_state.u_weight
        u_height = st.session_state.u_height
        u_age = st.session_state.u_age
        u_display_name = st.session_state.u_display_name
        
        # ... Rest of your Nutrition logic ...
    
    n_col1, n_col2 = st.columns(2)
    with n_col1:
        # Note: 'Mixed' is now an option for broader variety
        diet_type = st.selectbox("Dietary Preference", ["Vegetarian", "Eggitarian", "Non-Vegetarian", "Mixed (Everything)"])
    with n_col2:
        diet_goal = st.selectbox("Nutrition Goal", ["Muscle Gain", "Fat Loss", "Body Recomp"])
    
    special_requests = st.text_area("Allergies / Foods to Avoid / Any Other Preferences", placeholder="e.g. No peanuts, extra eggs,...")
    
    if st.button("🍳 Generate Diet Plan", use_container_width=True):
        with st.spinner("Calculating macros and designing menu..."):
            # 1. Use your function to get the math right
            maint, target = calculate_macros(u_weight, u_height, u_age, diet_goal)
            stats = {'target': target, 'maint': maint}
            
            # 2. Pass those stats into the AI function
            plan = generate_diet_only_plan(
                u_display_name, 
                diet_type, 
                diet_goal, 
                special_requests, 
                stats, 
                client
            )
            
            # 3. Save to the 'Vault' (Session State)
            st.session_state.current_diet = plan
            st.session_state.diet_stats = stats
            st.success("Plan Generated!")

    # 4. Display Logic with Safety Check
    if "current_diet" in st.session_state and "diet_stats" in st.session_state:
        d_stats = st.session_state.diet_stats
        
        # Display the math derived from your calculate_macros function
        st.info(f"🎯 Target: {d_stats['target']} kcal | ⚖️ Maintenance: {d_stats['maint']} kcal")
        
        st.markdown(st.session_state.current_diet)
        
        st.divider()
        try:
            diet_pdf_bytes = export_pdf(st.session_state.current_diet, u_display_name)
            st.download_button(
                label="📥 Download Nutrition Plan (PDF)",
                data=diet_pdf_bytes,
                file_name=f"{u_display_name}_Kinetix_Diet_Plan.pdf",
                mime="application/pdf",
                use_container_width=True,
                key="diet_export_btn" 
            )
        except Exception as e:
            st.error(f"PDF Error: {e}")


tracker_icon_data = base64.b64encode(open("tracker_icon.png", "rb").read()).decode()
with tab_tracker:
    # --- BRANDED TRACKER HEADER (Left-Aligned) ---
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
            <img src="data:image/png;base64,{tracker_icon_data}" width="50">
            <h1 style="background: linear-gradient(90deg, #FDC830, #F37335, #FF5F6D); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight: 800; margin: 0; line-height: 1;">Daily Meal Log</h1>
        </div>
        <hr style="margin-top: 2px; margin-bottom: 25px; border: 0; border-top: 2px solid #eee; opacity: 0.4;">
        """,
        unsafe_allow_html=True
    )
    
    # 1. SETUP: Select Number of Meals
    num_meals = st.selectbox("How many meals did you have today?", range(1, 7), index=2)
    
    # Initialize session state for meal data if it doesn't exist
    if "meal_data" not in st.session_state:
        st.session_state.meal_data = {}

    col_inputs, col_viz = st.columns([1, 1.2])

    with col_inputs:
        st.markdown("##### 🍽️ Log Each Meal")
        
        # This dictionary tracks the SUM of all analyzed meals for the chart
        temp_total_macros = {"Protein": 0, "Carbs": 0, "Fats": 0, "Calories": 0}
        
        # 2. DYNAMIC INPUTS: Generate boxes based on num_meals
        for i in range(1, num_meals + 1):
            meal_key = f"meal_{i}"
            
            # Input Box
            meal_input = st.text_input(f"Meal {i}:", placeholder="Enter Meal or Ingredients", key=f"input_{meal_key}")
            
            if st.button(f"🔍 Analyze Meal {i}", key=f"btn_{meal_key}"):
                if meal_input:
                    with st.spinner(f"Analyzing Meal {i}..."):
                        macros = get_macros_from_text(meal_input, client)
                        if macros:
                            st.session_state.meal_data[meal_key] = macros
                            st.success(f"Meal {i} updated!")
                        else:
                            st.error("Could not parse meal.")

            # --- THE BREAKDOWN LOGIC ---
            if meal_key in st.session_state.meal_data:
                m = st.session_state.meal_data[meal_key]
                
                # Individual Breakdown Expander
                with st.expander(f"📋 View Meal {i} Details", expanded=True):
                    # 1. Centered Total Calories Header
                    st.markdown(
                        f"""
                        <div style="text-align: center;">
                            <p style="margin-bottom: 0px; color: gray; font-size: 14px;">Total Meal Calories</p>
                            <h2 style="margin-top: 0px; color: #00D1FF;">{m['Calories']} kcal</h2>
                        </div>
                        """, 
                        unsafe_allow_html=True
                    )
                    
                    # 2. Aligned Macros below
                    st.divider()
                    b_col1, b_col2, b_col3 = st.columns(3)
                    
                    with b_col1:
                        st.markdown(f"<div style='text-align: center;'><b>Protein</b><br>{m['Protein']}g</div>", unsafe_allow_html=True)
                    with b_col2:
                        st.markdown(f"<div style='text-align: center;'><b>Carbs</b><br>{m['Carbs']}g</div>", unsafe_allow_html=True)
                    with b_col3:
                        st.markdown(f"<div style='text-align: center;'><b>Fats</b><br>{m['Fats']}g</div>", unsafe_allow_html=True)
                
                # Add this meal's macros to our total counter for the Pie Chart
                for k in temp_total_macros:
                    temp_total_macros[k] += m.get(k, 0)

    with col_viz:
        st.markdown("##### 📊 Daily Summary")
        
        # Only show the chart if at least one meal has been analyzed
        if any(v > 0 for v in temp_total_macros.values()):
            import plotly.graph_objects as go
            
            labels = ['Protein', 'Carbs', 'Fats']
            values = [temp_total_macros['Protein'], temp_total_macros['Carbs'], temp_total_macros['Fats']]
            
            fig = go.Figure(data=[go.Pie(
                labels=labels, 
                values=values, 
                hole=.4,
                marker=dict(colors=['#00D1FF', '#7000FF', '#FF007A'])
            )])
            fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=300, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            # Total Calories Metric
            st.metric("Total Daily Intake", f"{temp_total_macros['Calories']} kcal")
            
            # Progress Bars (You can adjust the /150, /200 targets based on the user's plan later)
            st.progress(min(temp_total_macros['Protein']/150, 1.0), text=f"Protein Progress: {temp_total_macros['Protein']}g")
            st.progress(min(temp_total_macros['Carbs']/200, 1.0), text=f"Carbs Progress: {temp_total_macros['Carbs']}g")
        else:
            st.info("Log and analyze your meals to see your daily breakdown here.")

    # Reset Button to clear the day
    if st.button("🗑️ Reset All Meal Logs", use_container_width=True):
        st.session_state.meal_data = {}
        st.rerun()

exercise_icon_data = base64.b64encode(open("exercise_icon.png", "rb").read()).decode()
with tab_exercise:
    # --- CORE AI: GREEN GRADIENT HEADER ---
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
            <img src="data:image/png;base64,{exercise_icon_data}" width="55">
            <h1 style="background: linear-gradient(90deg, #FDC830, #F37335, #FF5F6D); -webkit-background-clip: text; -webkit-text-fill-color: transparent; font-size: 2.8rem; font-weight: 800; margin: 0; line-height: 1;">Exercise Analyzer</h1>
        </div>
        <hr style="margin-top: 2px; margin-bottom: 25px; border: 0; border-top: 2px solid #eee; opacity: 0.4;">
        """,
        unsafe_allow_html=True
    )

    with st.expander("📌 MANDATORY FILMING INSTRUCTIONS", expanded=True):
        st.warning("""
        For accurate Analysis, please ensure:
        1. **Side Angle:** Camera must be placed 90° to your side.
        2. **Full Body:** Head, Hips, and Feet must be visible throughout the rep.
        3. **Lighting:** Ensure your joints are not obscured.
        4. **Correct Exercise:** Ensure the selected exercise matches your video.
        """)
    
    ex_type = st.selectbox("Select Exercise", ["Squat", "Deadlift", "Bench Press", "Shoulder Press"])
    video_upload = st.file_uploader("Upload video", type=['mp4', 'mov'])
    
    if video_upload:
        # 1. Prepare the video - Indented to be inside 'if video_upload'
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
        tfile.write(video_upload.read())
        path = tfile.name
        tfile.close() 
        
        ui_col1, ui_col2, ui_col3 = st.columns([1, 2, 1])
        with ui_col2:
            st.info("Original Upload")
            st.video(path)
        
        # 2. THE UPDATED BUTTON LOGIC - Indented to be inside 'if video_upload'
        if st.button("🚀 Analyze Form", key="final_analysis_trigger"):
            with st.spinner("Analyzing Video..."):
                feedback = process_video_locally(path, ex_type)
                
                res_col1, res_col2, res_col3 = st.columns([1, 2, 1])
                
                with res_col2:
                    if os.path.exists("final_output.mp4"):
                        st.subheader("🎥 Video Analysis Replay")
                        st.video("final_output.mp4")
                    
                    if os.path.exists("audit_result.jpg"):
                        st.subheader("📸 Key Movement Snapshot")
                        st.image("audit_result.jpg", use_container_width=True)
                
                st.divider()
                st.markdown(feedback)
                
                # --- SAVE TO DATABASE ---
                db = SessionLocal()
                db.add(ExerciseAudit(
                    user_id=st.session_state.user_id, 
                    exercise_name=ex_type, 
                    feedback_text=feedback
                ))
                db.commit()
                db.close()
                
                # CLEAN UP
                if os.path.exists(path):
                    os.remove(path)


with tab_history:
    # --- BRANDED HISTORY HEADER (Left-Aligned) ---
    st.markdown(
        f"""
        <div style="display: flex; align-items: center; gap: 15px; margin-bottom: 10px;">
            <img src="data:image/png;base64,{base64.b64encode(open("history_icon.png", "rb").read()).decode()}" width="50">
            <h1 style="
                background: linear-gradient(90deg, #FDC830, #F37335, #FF5F6D);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                font-size: 2.8rem;
                font-weight: 800;
                margin: 0;
                line-height: 1;
            ">{st.session_state.username}'s History</h1>
        </div>
        <hr style="margin-top: 2px; margin-bottom: 25px; border: 0; border-top: 2px solid #eee; opacity: 0.4;">
        """,
        unsafe_allow_html=True
    )
    
    # ... Rest of your history/database retrieval logic ...
    
    # --- 1. THE CLEAR HISTORY BUTTON ---
    col_a, col_b = st.columns([5, 1])
    with col_b:
        st.markdown('<div class="clear-btn">', unsafe_allow_html=True)
        if st.button("🗑️ Clear All", help="Permanently delete all saved plans and audits", use_container_width=True):
            # All logic below is now inside the button click
            db = SessionLocal()
            try:
                db.query(FitnessPlan).filter(FitnessPlan.user_id == st.session_state.user_id).delete()
                db.commit()
                st.success("History Deleted.")
                time.sleep(1)
                st.rerun()
            except Exception as e:
                st.error(f"Error clearing history: {e}")
            finally:
                db.close()
        st.markdown('</div>', unsafe_allow_html=True)

    # --- 2. DISPLAY SAVED PLANS ---
    st.subheader("📋 Saved Fitness Plans")
    db = SessionLocal()
    history = db.query(FitnessPlan).filter(FitnessPlan.user_id == st.session_state.user_id).order_by(FitnessPlan.id.desc()).all()    
    if history:
        for p in history:
            # Using a cleaner expander label
            with st.expander(f"✨ Goal: {p.goal.upper()} (Ref: #{p.id})"):
                st.markdown(p.plan_text)
                st.caption(f"Stored in local database session.")
    else:
        st.info("No fitness plans saved yet. Generate one in the 'Plan' tab!")    
    db.close()
