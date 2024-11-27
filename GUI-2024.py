#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tkinter as tk
from tkinter import messagebox, ttk
from tkinter.ttk import Style
from joblib import load
import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import csv
import matplotlib.pyplot as plt

# Load the trained GB model
model_path = '13-results/GB_best_model(Mtest).joblib'  # Adjust path as needed
gb_model = load(model_path)

# Feature names and tooltips
feature_names = [
    "Area of tensile reinforcement, As (mm²):",
    "The 0.2 proof strength, σ0.2 (MPa):",
    "Strain hardening parameter, n:",
    "Concrete compressive strength, fc (MPa):",
    "Beam width, b (mm):",
    "Effective depth, d (mm):",
]
tooltips = [
    "The area of steel reinforcement in tension (mm²).",
    "The width of the beam (mm).",
    "The effective depth from the top of the beam to the center of tension reinforcement (mm).",
    "The compressive strength of concrete (MPa).",
    "The 0.2 proof stress of material (MPa).",
    "Strain hardening parameter (dimensionless)."
]

# Setup for UI
root = tk.Tk()
root.title("Flexural Capacity Prediction")
root.geometry("480x700")
root.configure(bg='#F0F4F8')  # Lighter background color
root.tk.call('tk', 'scaling', 2.0)  # Set high DPI for better clarity

# Title label
main_title = tk.Label(root, text="Ultimate Flexural Capacity Prediction", font=('Times New Roman', 18, 'bold'), bg='#F0F4F8', fg='#34495E')
main_title.pack(pady=(10, 5))

# Style customization
style = ttk.Style()
style.theme_use('clam')
style.configure("Custom.TLabelframe", background='#F0F4F8', foreground='#34495E', font=('Times New Roman', 14, 'bold'))
style.configure("Custom.TButton", background='#3498DB', foreground='#FFFFFF', font=('Times New Roman', 12, 'bold'))
style.configure("Custom.TLabel", background='#F0F4F8', foreground='#34495E', font=('Times New Roman', 12))
style.configure("Dark.TLabelframe", background='#2C3E50', foreground='#ECF0F1')

# Frames
feature_frame = ttk.LabelFrame(root, text="Feature Inputs", padding=10, style="Custom.TLabelframe")
output_frame = ttk.LabelFrame(root, text="Model Output", padding=10, style="Custom.TLabelframe")
feature_frame.pack(padx=10, pady=5, fill='x')
output_frame.pack(padx=10, pady=5, fill='x')

# Input fields
entries = [ttk.Entry(feature_frame, font=('Times New Roman', 12), width=12) for _ in feature_names]

for i, (label_text, entry, tooltip) in enumerate(zip(feature_names, entries, tooltips)):
    tk.Label(feature_frame, text=label_text, font=('Times New Roman', 12), bg='#F0F4F8', fg='#34495E').grid(row=i, column=0, sticky='w', padx=5, pady=3)
    entry.grid(row=i, column=1, padx=5, pady=3, sticky='w')

# Output
output_label = tk.Label(output_frame, text="Mtest (kN·m): ", font=('Times New Roman', 14), bg='#F0F4F8', fg='#34495E')
output_label.grid(row=0, column=0, padx=5, pady=5, sticky='w')
output_widget = tk.Label(output_frame, text="", font=('Times New Roman', 14, 'bold'), bg='#F0F4F8', fg='#3498DB')
output_widget.grid(row=0, column=1, padx=5, pady=5, sticky='w')

# Prediction history listbox
history_frame = ttk.LabelFrame(root, text="Prediction History", padding=10, style="Custom.TLabelframe")
history_frame.pack(padx=10, pady=5, fill='x')
history_listbox = tk.Listbox(history_frame, height=5, font=('Times New Roman', 12))
history_listbox.pack(fill='both', padx=5, pady=5)

# Functions
def submit():
    try:
        feature_values = [float(entry.get()) for entry in entries]
        scaled_input = scale_features(feature_values)
        result = gb_model.predict(scaled_input)[0]
        output_widget.config(text=f"{result:.2f}")
        history_listbox.insert(tk.END, f"Features: {feature_values} -> Mtest: {result:.2f}")
    except ValueError:
        messagebox.showerror("Input Error", "Please enter valid numeric values for all fields.")

def scale_features(feature_values):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df = pd.read_excel('D:/12database/13.xlsx', names=[name.split(",")[0] for name in feature_names] + ["Mtest"])
    X = df.drop(columns=["Mtest"])
    scaler.fit(X)
    return scaler.transform([feature_values])

def clear_fields():
    for entry in entries:
        entry.delete(0, tk.END)
    output_widget.config(text="")

def clear_history():
    history_listbox.delete(0, tk.END)
    messagebox.showinfo("Cleared", "Prediction history cleared.")

def save_results():
    with open("prediction_results.csv", "a", newline="") as file:
        writer = csv.writer(file)
        feature_values = [entry.get() for entry in entries]
        writer.writerow(feature_values + [output_widget.cget("text")])
    messagebox.showinfo("Saved", "Results saved to prediction_results.csv")

def plot_history():
    if history_listbox.size() == 0:
        messagebox.showinfo("No Data", "No predictions to plot.")
        return
    
    values = [float(item.split("-> Mtest: ")[-1]) for item in history_listbox.get(0, tk.END)]
    
    plt.figure(figsize=(5, 3))
    plt.plot(values, marker='o')
    plt.title("Prediction History")
    plt.xlabel("Prediction Count")
    plt.ylabel("Mtest (kN·m)")
    plt.grid()

    # Set x-axis to integer ticks
    plt.xticks(range(len(values)))  # Ensures x-axis labels are integers

    # Ensure all labels and elements are visible
    plt.tight_layout()

    # Save as high-quality JPG before displaying
    save_as_jpg = messagebox.askyesno("Save Plot", "Would you like to save this plot as a high-quality JPG?")
    if save_as_jpg:
        plt.savefig("prediction_history.jpg", format='jpg', dpi=300)
        messagebox.showinfo("Saved", "Plot saved as 'prediction_history.jpg' in the current directory.")

    # Display the plot after saving
    plt.show()


def toggle_theme():
    if root.cget('bg') == '#F0F4F8':  # Light mode
        root.config(bg='#2C3E50')
        main_title.config(bg='#2C3E50', fg='#ECF0F1')
        footer.config(bg='#2C3E50', fg='#ECF0F1')
        feature_frame.config(style="Dark.TLabelframe")
        output_frame.config(style="Dark.TLabelframe")
        history_frame.config(style="Dark.TLabelframe")
    else:  # Dark mode
        root.config(bg='#F0F4F8')
        main_title.config(bg='#F0F4F8', fg='#34495E')
        footer.config(bg='#F0F4F8', fg='#34495E')
        feature_frame.config(style="Custom.TLabelframe")
        output_frame.config(style="Custom.TLabelframe")
        history_frame.config(style="Custom.TLabelframe")

def show_help():
    help_text = ("This tool predicts the ultimate flexural capacity (Mtest) of a beam based on several input parameters.\n\n"
                 "1. Enter values for each input field.\n"
                 "2. Click 'Predict' to see the predicted Mtest value.\n"
                 "3. Use 'Clear' to reset fields, 'Save Results' to store predictions, and 'Plot History' to visualize past predictions.\n\n"
                 "For further help, contact: sina.srfz@gmail.com.")
    messagebox.showinfo("Help", help_text)

# Buttons
button_frame = tk.Frame(root, bg='#F0F4F8')
button_frame.pack(pady=5)
predict_button = ttk.Button(button_frame, text="Predict", command=submit, style="Custom.TButton")
clear_button = ttk.Button(button_frame, text="Clear", command=clear_fields, style="Custom.TButton")
save_button = ttk.Button(button_frame, text="Save Results", command=save_results, style="Custom.TButton")
plot_button = ttk.Button(button_frame, text="Plot History", command=plot_history, style="Custom.TButton")
clear_history_button = ttk.Button(button_frame, text="Clear History", command=clear_history, style="Custom.TButton")
theme_button = ttk.Button(button_frame, text="Toggle Theme", command=toggle_theme, style="Custom.TButton")
help_button = ttk.Button(button_frame, text="Help", command=show_help, style="Custom.TButton")

# Button Layout
predict_button.grid(row=0, column=0, padx=5, pady=5)
clear_button.grid(row=0, column=1, padx=5, pady=5)
save_button.grid(row=0, column=2, padx=5, pady=5)
plot_button.grid(row=0, column=3, padx=5, pady=5)
clear_history_button.grid(row=0, column=4, padx=5, pady=5)
theme_button.grid(row=0, column=5, padx=5, pady=5)
help_button.grid(row=0, column=6, padx=5, pady=5)

# Footer
footer = tk.Label(root, text="Developed by Sina Sarfarazi - University of Naples Federico II, Italy", font=('Times New Roman', 10), bg='#F0F4F8', fg='#34495E')
footer.pack(pady=(5, 10))

# Run the main application loop
root.mainloop()


# In[ ]:




