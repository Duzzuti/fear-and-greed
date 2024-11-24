import matplotlib.pyplot as plt
import pandas as pd

class Graph:
    def __init__(self, df : pd.DataFrame, df_label="Graph", df_color="purple", df_line_width=1, y_axis="left"):
        self.df = df
        self.df_label = df_label
        self.df_color = df_color
        self.df_line_width = df_line_width
        self.y_axis = y_axis


def plot_graph(graph_list : list[Graph], title="", xlabel="Date", ylabel="Value", neutral_line=50):    
    # Plot the Fear and Greed Index
    plt.figure(figsize=(12, 6))
    left_y_axis = plt.gca()
    right_y_axis = left_y_axis.twinx()
    for graph in graph_list:
        # plot on the right y axis
        if graph.y_axis == "right":
            right_y_axis.plot(graph.df.index, graph.df, label=graph.df_label, color=graph.df_color, linewidth=graph.df_line_width)
        else:
            left_y_axis.plot(graph.df.index, graph.df, label=graph.df_label, color=graph.df_color, linewidth=graph.df_line_width)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if neutral_line is not None:
        plt.axhline(neutral_line, color='grey', linestyle='--', label="Neutral")
        # Add legends for both axes
    lines_left, labels_left = left_y_axis.get_legend_handles_labels()
    lines_right, labels_right = right_y_axis.get_legend_handles_labels()

    # Combine and add the legends
    plt.legend(lines_left + lines_right, labels_left + labels_right, loc='upper left')
    plt.show()
