from datetime import datetime, timedelta

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def plot_workflow(lineage_data):

    # Helper function to clean and parse timestamps
    def clean_and_parse_isoformat(timestamp):
        if "+" in timestamp and len(timestamp.split("+")[-1]) < 5:
            # Fix UTC offset formatting if it's incorrect
            timestamp = timestamp.split("+")[0] + "+00:00"
        return datetime.fromisoformat(timestamp)

    # Parse timestamps and prepare data
    start_times = [
        clean_and_parse_isoformat(step["timestamp"]) for step in lineage_data
    ]
    stop_times = [clean_and_parse_isoformat(step["stop_time"]) for step in lineage_data]
    reasoners = sorted(
        list({step["reasoner_id"] for step in lineage_data}),
        key=lambda r: min(
            start_times[i]
            for i, step in enumerate(lineage_data)
            if step["reasoner_id"] == r
        ),
    )  # Unique reasoners sorted by first appearance

    reasoner_indices = {reasoner: i for i, reasoner in enumerate(reasoners)}

    # Assign unique colors to each reasoner
    import matplotlib.pyplot as plt

    color_map = plt.colormaps.get_cmap("tab20")
    reasoner_colors = {
        reasoner: f"rgba({int(c[0]*255)}, {int(c[1]*255)}, {int(c[2]*255)}, 0.85)"
        for c, reasoner in zip(color_map(range(len(reasoners))), reasoners)
    }

    # Calculate dynamic figure height based on the number of reasoners
    bar_height = 0.4  # Thicker bars
    num_reasoners = len(reasoners)
    figure_height = 600 + num_reasoners * 40  # Dynamic height

    # Define uniform grid for x-axis (time)
    time_start = min(start_times)
    time_end = max(stop_times)
    grid_interval = timedelta(seconds=5)  # 5-second intervals
    grid_times = [
        time_start + i * grid_interval
        for i in range((time_end - time_start) // grid_interval + 2)
    ]

    # Create the figure
    fig = go.Figure()

    # Plot bars for each reasoner
    for i, step in enumerate(lineage_data):
        reasoner = step["reasoner_id"]
        start = start_times[i].timestamp()
        stop = stop_times[i].timestamp()
        duration = stop - start
        y_pos = reasoner_indices[reasoner]
        # Add rectangles (boxes) for better visualization
        fig.add_shape(
            type="rect",
            x0=start,
            x1=stop,
            y0=y_pos - bar_height / 2,
            y1=y_pos + bar_height / 2,
            fillcolor=reasoner_colors[reasoner],
            line=dict(width=1, color="black"),
        )
        # Add hover text with duration and reasoner information
        fig.add_trace(
            go.Scatter(
                x=[start, stop],
                y=[y_pos, y_pos],
                mode="lines",
                line=dict(color=reasoner_colors[reasoner], width=0),  # Full area hover
                hoverinfo="text",
                text=f"Reasoner: {reasoner}<br>Start: {datetime.fromtimestamp(start).strftime('%H:%M:%S')}<br>Duration: {duration:.2f}s",
                name=reasoner,
            )
        )

    # Add horizontal separation lines
    for idx in range(len(reasoners)):
        fig.add_shape(
            type="line",
            x0=time_start.timestamp(),
            x1=time_end.timestamp(),
            y0=idx,
            y1=idx,
            line=dict(color="gray", width=1, dash="dot"),
        )

    # Format axes
    fig.update_layout(
        title="Workflow Execution Timeline by Reasoner",
        xaxis=dict(
            title="Time (hh:mm:ss)",
            tickvals=[t.timestamp() for t in grid_times],
            ticktext=[t.strftime("%H:%M:%S") for t in grid_times],
            showgrid=True,
            zeroline=False,
        ),
        yaxis=dict(
            title="Reasoners",
            tickvals=list(range(len(reasoners))),
            ticktext=[
                i.replace("_", " ") for i in reasoners
            ],  # Replace underscores with spaces
            showgrid=False,
        ),
        height=figure_height,
        showlegend=False,  # Removed legend as reasoners are labeled in y-axis
        plot_bgcolor="white",
        margin=dict(l=150, r=50, t=50, b=100),
    )

    return fig


def plot_workflow_matplotlib(lineage_data, legend=False):
    # Helper function to clean and parse timestamps
    def clean_and_parse_isoformat(timestamp):
        if "+" in timestamp and len(timestamp.split("+")[-1]) < 5:
            # Fix UTC offset formatting if it's incorrect
            timestamp = timestamp.split("+")[0] + "+00:00"
        return datetime.fromisoformat(timestamp)

    # Parse timestamps and prepare data
    start_times = [
        clean_and_parse_isoformat(step["timestamp"]) for step in lineage_data
    ]
    stop_times = [clean_and_parse_isoformat(step["stop_time"]) for step in lineage_data]

    # Sort reasoners by their first start time
    reasoners = sorted(
        list({step["reasoner_id"] for step in lineage_data}),
        key=lambda r: min(
            start_times[i]
            for i, step in enumerate(lineage_data)
            if step["reasoner_id"] == r
        ),
    )
    reasoner_indices = {reasoner: i for i, reasoner in enumerate(reasoners)}

    # Assign unique colors to each reasoner
    color_map = plt.colormaps.get_cmap("tab10")
    reasoner_colors = {
        reasoner: color_map(i / len(reasoners)) for i, reasoner in enumerate(reasoners)
    }

    # Calculate dynamic figure height based on the number of reasoners
    bar_height = 0.2  # Standard thickness for all bars
    num_reasoners = len(reasoners)
    figure_height = max(5, num_reasoners * 0.6)  # Dynamic height, minimum 5

    # Define uniform grid for x-axis (time)
    time_start = min(start_times)
    time_end = max(stop_times)
    grid_interval = timedelta(seconds=5)  # 5-second intervals
    grid_times = [
        time_start + i * grid_interval
        for i in range((time_end - time_start) // grid_interval + 2)
    ]

    # Plot the timeline
    fig = plt.figure(figsize=(14, figure_height))

    # Plot bars for each reasoner
    for i, step in enumerate(lineage_data):
        reasoner = step["reasoner_id"]
        start = start_times[i].timestamp()
        stop = stop_times[i].timestamp()
        duration = stop - start
        plt.barh(
            reasoner_indices[reasoner],
            duration,
            left=start,
            height=bar_height,
            color=reasoner_colors[reasoner],
            edgecolor="black",
        )
        plt.text(
            start + duration / 2,
            reasoner_indices[reasoner],
            f"{duration:.2f}s",
            va="center",
            ha="center",
            fontsize=8,
        )

    # Add horizontal lines for separation
    for idx in range(len(reasoners)):
        plt.axhline(idx, color="gray", linestyle="--", linewidth=0.5, alpha=0.7)

    # Add legend for reasoner colors
    handles = [
        mpatches.Patch(color=color, label=reasoner)
        for reasoner, color in reasoner_colors.items()
    ]
    if legend:
        plt.legend(
            handles=handles,
            title="Reasoners",
            loc="lower center",
            bbox_to_anchor=(0.5, -0.5),
            ncol=3,
            fontsize=10,
            title_fontsize=12,
            frameon=False,
        )

    # Format axes
    plt.yticks(range(len(reasoners)), [i.replace("_", " ") for i in reasoners])
    plt.xticks(
        [t.timestamp() for t in grid_times],
        [t.strftime("%H:%M:%S") for t in grid_times],
        rotation=45,
        ha="right",
    )
    plt.xlabel("Time (hh:mm:ss)", labelpad=20)  # Increase label padding
    plt.ylabel("Reasoners")
    plt.title("Multi-Agent Workflow")

    # Adjust layout to avoid overlap
    if legend:
        plt.subplots_adjust(
            bottom=0.5
        )  # Increase bottom margin for the xlabel and legend
    plt.tight_layout()
    return fig
