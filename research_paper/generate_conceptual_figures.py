import os
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

# Apply styling globally
plt.style.use('seaborn-v0_8-paper')

def draw_box(ax, x, y, width, height, text, color, text_color="white", fontsize=11):
    box_patch = mpatches.FancyBboxPatch((x - width/2, y - height/2), width, height,
        boxstyle="round,pad=0.2", fc=color, ec="white", lw=2, zorder=3)
    ax.add_patch(box_patch)
    ax.text(x, y, text, ha="center", va="center", color=text_color, 
            fontsize=fontsize, fontweight="bold", zorder=4)

def draw_arrow(ax, x1, y1, x2, y2, label="", rad=0.0):
    ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle="-|>", color="#444444", lw=2.5, connectionstyle=f"arc3,rad={rad}"), zorder=2)
    if label:
        ax.text((x1+x2)/2, (y1+y2)/2 + 0.35, label, ha="center", va="center", 
                fontsize=10, color="#111111", fontweight="bold", zorder=4,
                bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1))

def draw_bubble(ax, x, y, radius, text, color, text_color="white"):
    circle = plt.Circle((x, y), radius, color=color, ec='white', lw=3, zorder=3)
    ax.add_patch(circle)
    ax.text(x, y, text, ha='center', va='center', color=text_color, fontsize=11, fontweight='bold', zorder=4)

def generate_data_flow():
    fig, ax = plt.subplots(figsize=(10, 7))
    ax.axis("off")
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 7)
    
    # Bubbles for Data Sources
    draw_bubble(ax, 2, 5.2, 1.25, "BTS Data\n(US DOT)\nHistorical delays", "#1565C0")
    draw_bubble(ax, 2, 1.8, 1.25, "NOAA Feeds\n(METAR)\nReal-time weather", "#00838F")
    
    # Process box
    draw_box(ax, 5.5, 3.5, 2.8, 1.8, "Preprocessing\nMapping Engine", "#4527A0", fontsize=12)
    
    # Output bubble
    draw_bubble(ax, 9.2, 3.5, 1.35, "Merged Vector\n(110 Columns)\nReady for ML", "#2E7D32")
    
    # Arrows
    draw_arrow(ax, 3.25, 5, 4.4, 4.0, label="Extract & Clean")
    draw_arrow(ax, 3.25, 2, 4.4, 3.0, label="Live Interpolate")
    draw_arrow(ax, 6.9, 3.5, 7.85, 3.5, label="Export Data")
    
    # Annotation on the merge
    ax.text(5.5, 6, "Merge Logic (Spatial-Temporal Threshold):\nDistance Lat/Long <= 0.1°\nTime proximity <= 60 mins", 
            ha="center", va="center", bbox=dict(boxstyle="round,pad=0.5", fc="#FFF3E0", ec="#FFB74D", lw=2), 
            fontsize=11, fontweight='bold', color="#E65100", zorder=5)
            
    # Connect threshold
    ax.plot([5.5, 5.5], [5.2, 4.4], ls='--', color="#FFB74D", lw=3, zorder=2)
    
    plt.title("Figure 3: Data Flow (BTS & NOAA Merge Integration)", fontsize=16, fontweight="bold", pad=15)
    
    os.makedirs('figures', exist_ok=True)
    out_path = 'figures/fig3_data_flow_bubble_diagram.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved -> {out_path}")
    

def generate_architecture():
    fig, ax = plt.subplots(figsize=(11, 4.5))
    ax.axis("off")
    ax.set_xlim(0, 12)
    ax.set_ylim(-0.5, 5)
    
    # Tier 1
    draw_box(ax, 2, 2.5, 2.8, 3.5, "Tier 1: Modeling\n\n• XGBoost Engine\n• Optuna HPO\n• Scikit-learn Pipeline", "#283593")
    
    # Tier 2
    draw_box(ax, 6, 2.5, 2.8, 3.5, "Tier 2: Backend\n\n• FastAPI Service\n• REST Endpoints\n• FlightTracker Worker", "#00695C")
    
    # Tier 3
    draw_box(ax, 10, 2.5, 2.8, 3.5, "Tier 3: Frontend\n\n• Next.js Dashboard\n• React Interactive UI\n• Map Visualizations", "#E65100")
    
    # Connections
    draw_arrow(ax, 3.4, 3.2, 4.6, 3.2, label="Model Weights (.pkl)")
    draw_arrow(ax, 7.4, 3.2, 8.6, 3.2, label="JSON Output / API")
    
    # Reverse arrow for user queries
    ax.annotate("", xy=(7.4, 1.8), xytext=(8.6, 1.8),
                arrowprops=dict(arrowstyle="-|>", color="#B71C1C", lw=2.5, ls="--"), zorder=2)
    ax.text(8, 1.2, "User Search Queries", ha="center", fontsize=9, color="#B71C1C", fontweight="bold",
            bbox=dict(facecolor='white', edgecolor='none', alpha=0.9, pad=1))
    
    plt.title("Figure 4: SkyPredict Three-Tier Technical Architecture", fontsize=16, fontweight="bold", pad=5)
    
    out_path = 'figures/fig4_three_tier_architecture.png'
    plt.savefig(out_path, dpi=300, bbox_inches='tight')
    print(f"Saved -> {out_path}")

if __name__ == "__main__":
    print("--- Generating Conceptual Architectures ---")
    generate_data_flow()
    generate_architecture()
    print("--- Done ---")
