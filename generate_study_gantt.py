"""
Study Agenda Gantt Chart Generator
===================================
Generates an interactive HTML Gantt chart showing the 28-week study plan.

Quick Test:
    python generate_study_gantt.py
    # Opens study_gantt.html in browser

Dependencies:
    pip install plotly pandas
"""

from pathlib import Path
import webbrowser


def create_gantt_data():
    """
    Define the study schedule data.
    Each entry: (Section, Subsection, Start Week, End Week, Activity Type)
    Activity Types: Learn, Review, Project, Mock, DSA
    """
    
    schedule = [
        # Phase 0: Time Series (Weeks 1-2)
        ("4. Time Series", "4.3 Modern Methods", 1, 1, "Learn"),
        ("4. Time Series", "4.4 Practical Considerations", 1, 1, "Learn"),
        ("4. Time Series", "4.5 Uncertainty Quantification", 2, 2, "Learn"),
        ("4. Time Series", "4.1-4.2 Review", 2, 2, "Review"),
        
        # Phase 1: Stats + ML Foundations (Weeks 3-7)
        ("2. Stats", "2.1 Probability Basics", 3, 3, "Learn"),
        ("2. Stats", "2.2 Distributions", 4, 4, "Learn"),
        ("2. Stats", "2.3 Statistical Inference", 4, 4, "Learn"),
        ("2. Stats", "2.4 Advanced Theory", 5, 5, "Learn"),
        ("3. ML/DL", "3.1.1-3.1.2 Core + Linear", 4, 4, "Learn"),
        ("3. ML/DL", "3.1.3-3.1.4 Trees + Eval", 5, 5, "Learn"),
        ("3. ML/DL", "3.1.5-3.1.6 SVM + Imbalanced", 6, 6, "Learn"),
        ("9. Breadth", "9.3 Survival Analysis", 6, 6, "Learn"),
        ("2. Stats", "Stats + ML Review", 7, 7, "Review"),
        ("3. ML/DL", "Stats + ML Review", 7, 7, "Review"),
        ("Mock Interview", "#1 Stats + ML", 7, 7, "Mock"),
        
        # Phase 2: Causal Deep Dive (Weeks 8-13)
        ("1. Causal", "1.1 Foundations", 8, 8, "Learn"),
        ("1. Causal", "1.2.1-1.2.3 A/B, DiD, RDD", 9, 9, "Learn"),
        ("1. Causal", "1.2.4-1.2.5 IV, Synthetic", 10, 10, "Learn"),
        ("1. Causal", "1.2.6-1.2.7 PSM, Bandits", 11, 11, "Learn"),
        ("1. Causal", "1.3 Modern Causal ML", 12, 12, "Learn"),
        ("1. Causal", "1.4-1.5 Libraries + Sensitivity", 13, 13, "Learn"),
        ("Mock Interview", "#2 Causal Deep Dive", 13, 13, "Mock"),
        
        # Phase 3: DL + ML Depth + Project 1 (Weeks 14-18)
        ("3. ML/DL", "3.3.1-3.3.2 NN Fundamentals", 14, 14, "Learn"),
        ("3. ML/DL", "3.3.3-3.3.4 Architectures, Transfer", 15, 15, "Learn"),
        ("3. ML/DL", "3.2 Unsupervised", 16, 16, "Learn"),
        ("3. ML/DL", "3.4 Feature Engineering", 16, 16, "Learn"),
        ("3. ML/DL", "3.5 Tuning + Interpretability", 17, 17, "Learn"),
        ("1. Causal", "1.6 Applications", 18, 18, "Learn"),
        ("Project", "Project 1: Causal Uplift", 15, 18, "Project"),
        ("Mock Interview", "#3 Causal + ML", 18, 18, "Mock"),
        
        # Phase 4: NLP/GenAI + Project 2 (Weeks 19-23)
        ("5. NLP/GenAI", "5.1 NLP Fundamentals", 19, 19, "Learn"),
        ("5. NLP/GenAI", "5.2 Transformers & LLMs", 19, 19, "Learn"),
        ("5. NLP/GenAI", "5.3 RAG", 20, 20, "Learn"),
        ("5. NLP/GenAI", "5.4-5.5 Agentic AI, Causal+AI", 21, 21, "Learn"),
        ("9. Breadth", "9.1 HMM", 22, 22, "Learn"),
        ("9. Breadth", "9.2 RL Basics", 22, 22, "Learn"),
        ("7. Coding", "7.3 Best Practices", 23, 23, "Learn"),
        ("Project", "Project 2: TBD", 21, 23, "Project"),
        ("Mock Interview", "#4 GenAI + System", 23, 23, "Mock"),
        
        # Phase 5: System Design + Interview Sprint (Weeks 24-28)
        ("6. System Design", "6.1 Design Patterns", 24, 24, "Learn"),
        ("6. System Design", "6.2 Key Components", 24, 24, "Learn"),
        ("6. System Design", "6.3 Case Studies", 25, 25, "Learn"),
        ("7. Coding", "7.1 Core Patterns", 26, 26, "Learn"),
        ("7. Coding", "7.2 Python for ML", 26, 26, "Learn"),
        ("8. Product Sense", "8.1-8.3 Full Section", 27, 27, "Learn"),
        ("Mock Interview", "#5-8 Weekly Mocks", 24, 28, "Mock"),
        ("All Sections", "Final Review", 28, 28, "Review"),
        
        # Parallel activities (shown as continuous bars)
        ("DSA Practice", "Daily 30min", 1, 28, "DSA"),
    ]
    
    return schedule


def generate_html_gantt(schedule):
    """
    Generate a self-contained HTML file with the Gantt chart.
    Uses inline CSS and JavaScript for a completely standalone file.
    """
    
    # Define colors for activity types
    colors = {
        "Learn": "#4CAF50",      # Green
        "Review": "#2196F3",     # Blue
        "Project": "#FF9800",    # Orange
        "Mock": "#9C27B0",       # Purple
        "DSA": "#607D8B",        # Gray
    }
    
    # Group by section for better organization
    sections = {}
    for section, subsection, start, end, activity in schedule:
        if section not in sections:
            sections[section] = []
        sections[section].append((subsection, start, end, activity))
    
    # Calculate row positions
    row_height = 30
    row_gap = 5
    current_y = 60
    rows = []
    section_headers = []
    
    section_order = [
        "1. Causal", "2. Stats", "3. ML/DL", "4. Time Series",
        "5. NLP/GenAI", "6. System Design", "7. Coding", "8. Product Sense",
        "9. Breadth", "Project", "Mock Interview", "DSA Practice", "All Sections"
    ]
    
    for section in section_order:
        if section not in sections:
            continue
        section_headers.append((section, current_y))
        for subsection, start, end, activity in sections[section]:
            rows.append({
                "section": section,
                "subsection": subsection,
                "start": start,
                "end": end,
                "activity": activity,
                "y": current_y,
                "color": colors.get(activity, "#999"),
            })
            current_y += row_height + row_gap
        current_y += 10  # Extra gap between sections
    
    total_height = current_y + 50
    week_width = 35
    label_width = 250
    total_width = label_width + (28 * week_width) + 50
    
    # Generate SVG bars
    bars_svg = ""
    for row in rows:
        x = label_width + (row["start"] - 1) * week_width
        width = (row["end"] - row["start"] + 1) * week_width - 4
        bars_svg += f'''
        <g class="bar" data-section="{row['section']}" data-activity="{row['activity']}">
            <rect x="{x}" y="{row['y']}" width="{width}" height="{row_height - 2}" 
                  fill="{row['color']}" rx="4" class="bar-rect"/>
            <text x="{x + 5}" y="{row['y'] + 18}" class="bar-text" fill="white" font-size="11">
                {row['subsection'][:30]}{'...' if len(row['subsection']) > 30 else ''}
            </text>
        </g>'''
    
    # Generate section labels
    labels_svg = ""
    for section, y in section_headers:
        labels_svg += f'''
        <text x="10" y="{y + 18}" font-weight="bold" font-size="12" fill="#333">{section}</text>'''
    
    # Generate week headers
    weeks_svg = ""
    for week in range(1, 29):
        x = label_width + (week - 1) * week_width
        weeks_svg += f'''
        <text x="{x + week_width/2}" y="40" text-anchor="middle" font-size="10" fill="#666">W{week}</text>
        <line x1="{x}" y1="50" x2="{x}" y2="{total_height - 30}" stroke="#eee" stroke-width="1"/>'''
    
    # Phase markers
    phases = [
        (1, 2, "Phase 0: Time Series"),
        (3, 7, "Phase 1: Stats + ML"),
        (8, 13, "Phase 2: Causal"),
        (14, 18, "Phase 3: DL + Project 1"),
        (19, 23, "Phase 4: GenAI + Project 2"),
        (24, 28, "Phase 5: Interview Sprint"),
    ]
    
    phase_svg = ""
    for start, end, name in phases:
        x = label_width + (start - 1) * week_width
        width = (end - start + 1) * week_width
        phase_svg += f'''
        <rect x="{x}" y="5" width="{width}" height="20" fill="#f0f0f0" stroke="#ccc" rx="3"/>
        <text x="{x + width/2}" y="19" text-anchor="middle" font-size="9" fill="#555">{name}</text>'''
    
    html = f'''<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Study Agenda - 28 Week Gantt Chart</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 20px;
            background: #fafafa;
        }}
        h1 {{
            color: #333;
            margin-bottom: 10px;
        }}
        .legend {{
            display: flex;
            gap: 20px;
            margin-bottom: 20px;
            flex-wrap: wrap;
        }}
        .legend-item {{
            display: flex;
            align-items: center;
            gap: 5px;
            font-size: 13px;
        }}
        .legend-color {{
            width: 20px;
            height: 20px;
            border-radius: 4px;
        }}
        .chart-container {{
            overflow-x: auto;
            background: white;
            border: 1px solid #ddd;
            border-radius: 8px;
            padding: 10px;
        }}
        svg {{
            display: block;
        }}
        .bar-rect {{
            cursor: pointer;
            transition: opacity 0.2s;
        }}
        .bar-rect:hover {{
            opacity: 0.8;
            stroke: #333;
            stroke-width: 2;
        }}
        .bar-text {{
            pointer-events: none;
            font-family: inherit;
        }}
        .filter-buttons {{
            margin-bottom: 15px;
        }}
        .filter-buttons button {{
            padding: 8px 16px;
            margin-right: 8px;
            border: 1px solid #ddd;
            border-radius: 4px;
            background: white;
            cursor: pointer;
            font-size: 13px;
        }}
        .filter-buttons button:hover {{
            background: #f0f0f0;
        }}
        .filter-buttons button.active {{
            background: #333;
            color: white;
            border-color: #333;
        }}
        .stats {{
            margin-top: 20px;
            padding: 15px;
            background: #f5f5f5;
            border-radius: 8px;
            font-size: 14px;
        }}
        .stats h3 {{
            margin-top: 0;
        }}
    </style>
</head>
<body>
    <h1>Senior Applied Scientist Study Agenda</h1>
    <p>28 Weeks | 560 Hours | 20h/week</p>
    
    <div class="legend">
        <div class="legend-item"><div class="legend-color" style="background: #4CAF50"></div> Learn</div>
        <div class="legend-item"><div class="legend-color" style="background: #2196F3"></div> Review</div>
        <div class="legend-item"><div class="legend-color" style="background: #FF9800"></div> Project</div>
        <div class="legend-item"><div class="legend-color" style="background: #9C27B0"></div> Mock Interview</div>
        <div class="legend-item"><div class="legend-color" style="background: #607D8B"></div> DSA (Daily)</div>
    </div>
    
    <div class="filter-buttons">
        <button class="active" onclick="filterBars('all')">All</button>
        <button onclick="filterBars('Learn')">Learn</button>
        <button onclick="filterBars('Review')">Review</button>
        <button onclick="filterBars('Project')">Projects</button>
        <button onclick="filterBars('Mock')">Mocks</button>
    </div>
    
    <div class="chart-container">
        <svg width="{total_width}" height="{total_height}">
            <!-- Phase headers -->
            {phase_svg}
            
            <!-- Week headers -->
            {weeks_svg}
            
            <!-- Section labels -->
            {labels_svg}
            
            <!-- Bars -->
            {bars_svg}
        </svg>
    </div>
    
    <div class="stats">
        <h3>Summary Statistics</h3>
        <p><strong>Primary Spike:</strong> Causal Inference (~100h) - Weeks 8-13, 18</p>
        <p><strong>Secondary Spike:</strong> NLP/GenAI (~40h) - Weeks 19-21</p>
        <p><strong>Projects:</strong> 2 portfolio projects (Weeks 15-18, 21-23)</p>
        <p><strong>Mock Interviews:</strong> 7+ scheduled (Weeks 7, 13, 18, 23, 24-28)</p>
    </div>
    
    <script>
        function filterBars(activity) {{
            const bars = document.querySelectorAll('.bar');
            const buttons = document.querySelectorAll('.filter-buttons button');
            
            buttons.forEach(btn => btn.classList.remove('active'));
            event.target.classList.add('active');
            
            bars.forEach(bar => {{
                if (activity === 'all' || bar.dataset.activity === activity) {{
                    bar.style.opacity = '1';
                }} else {{
                    bar.style.opacity = '0.15';
                }}
            }});
        }}
    </script>
</body>
</html>'''
    
    return html


def main():
    """Generate the Gantt chart and open in browser."""
    print("Generating Study Agenda Gantt Chart...")
    
    schedule = create_gantt_data()
    html = generate_html_gantt(schedule)
    
    output_path = Path(__file__).parent / "study_gantt.html"
    output_path.write_text(html, encoding="utf-8")
    
    print(f"Gantt chart saved to: {output_path}")
    print("Opening in browser...")
    
    webbrowser.open(output_path.as_uri())


if __name__ == "__main__":
    main()
