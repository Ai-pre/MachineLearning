"""
Anime Recommender GUI (Using only 2 PKL files)
==============================================
Loads only:
  - models/svd_model.pkl
  - models/meta_model.pkl

No training required - instant startup!
"""

# -----------------------------------------------------------
# GUI / LIBRARY OVERVIEW
# -----------------------------------------------------------
# tkinter:
#   - Python's standard built-in GUI toolkit.
#   - Provides basic widgets and window management
#     (Tk(), Frame, Label, Button, Entry, etc.).
#
# import tkinter as tk:
#   - Imports the tkinter module and aliases it as "tk".
#   - All core widgets are then referenced as tk.WidgetName,
#     e.g., tk.Frame, tk.Label, tk.Button.
#
# from tkinter import ttk:
#   - Imports the "themed tk" (ttk) submodule.
#   - ttk provides modern styled widgets (Treeview, Combobox,
#     Progressbar, Notebook, etc.) that look better and more
#     native than classic tkinter widgets.
#
# from tkinter import messagebox:
#   - Provides simple pop-up dialog boxes for user interaction:
#       - messagebox.showinfo(title, message)
#       - messagebox.showwarning(title, message)
#       - messagebox.showerror(title, message)
#   - Used for notifications, warnings, and error messages.
#
# from tkinter import scrolledtext:
#   - Provides ScrolledText widget, a text box with a built-in
#     vertical scrollbar.
#   - Useful for logs / long texts without manually configuring
#     a separate Scrollbar widget.
#
# -----------------------------------------------------------

import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
import pandas as pd
import numpy as np
import pickle
import os


class AnimeRecommenderGUI:
    def __init__(self, root):
        """
        Initialize the main GUI application.

        Parameters
        ----------
        root : tk.Tk
            The root Tkinter window.
        """
        self.root = root
        self.root.title("Anime Recommendation System")
        self.root.geometry("1200x800")
        self.root.configure(bg="#f0f0f0")

        # Model-related variables
        self.svd_package = None        # Dictionary loaded from svd_model.pkl
        self.meta_package = None       # Dictionary loaded from meta_model.pkl
        self.svd_model = None          # Trained SVD model (e.g., Surprise SVD)
        self.meta_model = None         # Trained meta-learner model
        self.selected_model_type = tk.StringVar(value="Auto")  # Not used directly but reserved

        # Data-related variables
        self.meta_df = None            # DataFrame for meta features used during training
        self.anime_df = None           # DataFrame for anime metadata (anime.csv)
        self.ratings_df = None         # DataFrame for user ratings
        self.malid_to_idx = {}         # Mapping: MAL_ID -> index in item_sim_matrix
        self.idx_to_malid = {}         # Mapping: index in item_sim_matrix -> MAL_ID
        self.item_sim_matrix = None    # Precomputed item-item similarity matrix
        self.scaler = None             # Scaler used for meta model features (if needed)

        # Build GUI components and load models/data
        self.setup_ui()
        self.load_models()

    # =====================================================
    # GUI SETUP
    # =====================================================
    def setup_ui(self):
        """
        Create and arrange all GUI components.
        """
        # ---------------- Title Bar ----------------
        title_frame = tk.Frame(self.root, bg="#2c3e50", height=60)
        title_frame.pack(fill=tk.X)
        title_frame.pack_propagate(False)

        tk.Label(
            title_frame,
            text="Anime Recommendation System",
            font=("Helvetica", 24, "bold"),
            bg="#2c3e50",
            fg="white"
        ).pack(pady=10)

        # ---------------- Main Container ----------------
        main_frame = tk.Frame(self.root, bg="#f0f0f0")
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)

        # =================================================
        # LEFT PANEL: Controls + Logs
        # =================================================
        left_panel = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=2)
        left_panel.pack(side=tk.LEFT, fill=tk.BOTH, padx=(0, 10), ipadx=10, ipady=10)

        # ----- Loaded Model Info -----
        info_frame = tk.LabelFrame(
            left_panel,
            text="Loaded Model Info",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg="black"
        )
        info_frame.pack(fill=tk.X, padx=10, pady=10)

        self.model_info_label = tk.Label(
            info_frame,
            text="Loading...",
            font=("Helvetica", 9),
            bg="white",
            fg="black",
            justify=tk.LEFT,
            anchor=tk.W
        )
        self.model_info_label.pack(padx=10, pady=10, fill=tk.X)

        # ----- User-based Recommendations -----
        user_frame = tk.LabelFrame(
            left_panel,
            text="User-based Recommendations",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg="black"
        )
        user_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            user_frame,
            text="User ID:",
            font=("Helvetica", 10),
            bg="white",
            fg="black"
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Entry for target user ID
        self.user_entry = tk.Entry(user_frame, font=("Helvetica", 10), width=20)
        self.user_entry.pack(padx=10, pady=(0, 5))

        tk.Label(
            user_frame,
            text="Top N:",
            font=("Helvetica", 10),
            bg="white",
            fg="black"
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Spinbox to choose how many recommendations to display
        self.top_n_user = tk.Spinbox(user_frame, from_=5, to=50, font=("Helvetica", 10), width=18)
        self.top_n_user.delete(0, tk.END)
        self.top_n_user.insert(0, "10")
        self.top_n_user.pack(padx=10, pady=(0, 10))

        # Button to trigger user-based recommendation
        tk.Button(
            user_frame,
            text="Get Recommendations",
            command=self.get_user_recommendations,
            font=("Helvetica", 10, "bold"),
            bg="white",
            fg="black",
            activebackground="black",
            activeforeground="white",
            cursor="hand2",
            highlightthickness=0,
            highlightbackground="black",
            highlightcolor="black",
        ).pack(padx=10, pady=10)

        # ----- Similar Anime Recommendations -----
        anime_frame = tk.LabelFrame(
            left_panel,
            text="Similar Anime Recommendations",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg="black"
        )
        anime_frame.pack(fill=tk.X, padx=10, pady=10)

        tk.Label(
            anime_frame,
            text="Anime Title:",
            font=("Helvetica", 10),
            bg="white",
            fg="black"
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Entry for target anime title
        self.anime_entry = tk.Entry(anime_frame, font=("Helvetica", 10), width=20)
        self.anime_entry.pack(padx=10, pady=(0, 5))

        tk.Label(
            anime_frame,
            text="Top N:",
            font=("Helvetica", 10),
            bg="white",
            fg="black"
        ).pack(anchor=tk.W, padx=10, pady=(10, 5))

        # Spinbox for number of similar titles
        self.top_n_anime = tk.Spinbox(anime_frame, from_=5, to=50, font=("Helvetica", 10), width=18)
        self.top_n_anime.delete(0, tk.END)
        self.top_n_anime.insert(0, "10")
        self.top_n_anime.pack(padx=10, pady=(0, 10))

        # Button to trigger similar-anime search
        tk.Button(
            anime_frame,
            text="Find Similar Anime",
            command=self.get_similar_anime,
            font=("Helvetica", 10, "bold"),
            bg="white",
            fg="black",
            activebackground="black",
            activeforeground="white",
            cursor="hand2",
            highlightthickness=0,
            highlightbackground="black",
            highlightcolor="black",
        ).pack(padx=10, pady=10)

        # ----- System Log (ScrolledText) -----
        log_frame = tk.LabelFrame(
            left_panel,
            text="System Log",
            font=("Helvetica", 12, "bold"),
            bg="white",
            fg="black"
        )
        log_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # ScrolledText widget to display runtime logs/status messages
        self.log_text = scrolledtext.ScrolledText(
            log_frame,
            height=10,
            font=("Courier", 9),
            bg="#f8f9fa",
            fg="black",
            wrap=tk.WORD
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        # =================================================
        # RIGHT PANEL: Recommendation Results
        # =================================================
        right_panel = tk.Frame(main_frame, bg="white", relief=tk.RAISED, bd=2)
        right_panel.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)

        tk.Label(
            right_panel,
            text="Recommendations",
            font=("Helvetica", 16, "bold"),
            bg="white",
            fg="black"
        ).pack(pady=10)

        # Frame containing the Treeview and its scrollbar
        tree_frame = tk.Frame(right_panel, bg="white")
        tree_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=(0, 10))

        # Vertical scrollbar for Treeview
        scrollbar = ttk.Scrollbar(tree_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Treeview to show ranked recommendations
        self.tree = ttk.Treeview(
            tree_frame,
            columns=("Rank", "Title", "Score", "Info"),
            show="headings",
            yscrollcommand=scrollbar.set,
            height=20
        )
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.tree.yview)

        # Configure Treeview columns and headers
        self.tree.heading("Rank", text="Rank")
        self.tree.heading("Title", text="Anime Title")
        self.tree.heading("Score", text="Predicted Score")
        self.tree.heading("Info", text="Additional Info")

        self.tree.column("Rank", width=60, anchor=tk.CENTER)
        self.tree.column("Title", width=400, anchor=tk.W)
        self.tree.column("Score", width=120, anchor=tk.CENTER)
        self.tree.column("Info", width=200, anchor=tk.W)

        # ttk style configuration for a cleaner look
        style = ttk.Style()
        style.configure("Treeview", font=("Helvetica", 10), rowheight=25)
        style.configure("Treeview.Heading", font=("Helvetica", 11, "bold"))

    def log(self, message: str):
        """
        Append a log message to the System Log box and keep it scrolled.

        Parameters
        ----------
        message : str
            Text to append to the log.
        """
        self.log_text.insert(tk.END, f"{message}\n")
        self.log_text.see(tk.END)
        self.root.update()

    # =====================================================
    # LOAD MODELS
    # =====================================================
    def load_models(self):
        """
        Load SVD and Meta models from pickle files and load required CSV data.

        Expected file structure:
        - models/svd_model.pkl
        - models/meta_model.pkl
        - meta_train_ready.csv
        - anime.csv
        - rating_complete.csv
        """
        try:
            self.log("=" * 50)
            self.log("Loading models from pickle files...")
            self.log("=" * 50)

            # File paths for models
            svd_path = "models/svd_model.pkl"
            meta_path = "models/meta_model.pkl"

            # Check existence of required .pkl files
            if not os.path.exists(svd_path):
                raise FileNotFoundError(f"SVD model not found: {svd_path}")
            if not os.path.exists(meta_path):
                raise FileNotFoundError(f"Meta model not found: {meta_path}")

            # ----- Load SVD package -----
            self.log(f"Loading {svd_path}...")
            with open(svd_path, "rb") as f:
                self.svd_package = pickle.load(f)

            # Extract SVD model object
            self.svd_model = self.svd_package["model"]
            n_factors = self.svd_package.get("params", {}).get("n_factors", "Unknown")
            self.log(f"SVD model loaded (n_factors={n_factors})")

            # ----- Load Meta-Model package -----
            self.log(f"Loading {meta_path}...")
            with open(meta_path, "rb") as f:
                self.meta_package = pickle.load(f)

            # Extract components from meta package
            self.meta_model = self.meta_package["model"]
            self.scaler = self.meta_package.get("scaler", None)
            self.malid_to_idx = self.meta_package["malid_to_idx"]
            self.idx_to_malid = self.meta_package["idx_to_malid"]
            self.item_sim_matrix = self.meta_package["item_sim_matrix"]

            model_type = self.meta_package.get("model_type", "Unknown")
            rmse = self.meta_package.get("rmse", -1.0)
            self.log(f"Meta model loaded: {model_type} (RMSE={rmse:.4f})")

            # ----- Load CSV Data -----
            self.log("")
            self.log("Loading data files...")

            # Meta features used in training
            self.meta_df = pd.read_csv("meta_train_ready.csv")
            self.log(f"meta_train_ready.csv loaded ({len(self.meta_df)} records)")

            # Anime metadata
            self.anime_df = pd.read_csv("anime.csv")
            self.log(f"anime.csv loaded ({len(self.anime_df)} anime)")

            # User ratings
            self.ratings_df = pd.read_csv("rating_complete.csv")
            self.log(f"rating_complete.csv loaded ({len(self.ratings_df)} ratings)")

            # ----- Update model info label -----
            training_params = self.meta_package.get("training_params", {})
            sample_users = training_params.get("sample_users", 0)
            like_th = training_params.get("like_threshold", 0)

            info_text = (
                f"Model Type: {model_type}\n"
                f"RMSE: {rmse:.4f}\n"
                f"Anime Count: {len(self.malid_to_idx):,}\n"
                f"Users Sampled: {sample_users:,}\n"
                f"Like Threshold: {like_th}"
            )
            self.model_info_label.config(text=info_text)

            self.log("")
            self.log("=" * 50)
            self.log("ALL MODELS AND DATA LOADED SUCCESSFULLY.")
            self.log("=" * 50)

            # Inform the user via popup
            messagebox.showinfo(
                "Success",
                (
                    "Models loaded successfully.\n\n"
                    f"Meta Model: {model_type}\n"
                    f"RMSE: {rmse:.4f}\n"
                    "System is ready for recommendations."
                )
            )

        except FileNotFoundError as e:
            # Handle missing files
            self.log(f"\nERROR: {str(e)}")
            messagebox.showerror(
                "Files Not Found",
                (
                    f"{str(e)}\n\n"
                    "Please ensure the following files exist:\n"
                    "  - models/svd_model.pkl\n"
                    "  - models/meta_model.pkl"
                )
            )
        except Exception as e:
            # Handle any other load error
            self.log(f"\nERROR: {str(e)}")
            messagebox.showerror("Error", f"Failed to load models or data:\n{str(e)}")

    # =====================================================
    # CB SCORE COMPUTATION
    # =====================================================
    def get_cb_scores_for_user(self, user_id, like_th=8.0, top_k=30):
        """
        Compute a content-based (CB) score vector for a given user.

        Logic:
        - Find all anime that the user rated greater than or equal to like_th.
        - Map those anime IDs to indices in the item similarity matrix.
        - For each liked item, keep only its top_k most similar neighbors.
        - Aggregate (average) similarities across liked items to form a
          CB preference vector over all items.

        Parameters
        ----------
        user_id : int
            Target user ID.
        like_th : float
            Minimum rating threshold to consider an anime as "liked".
        top_k : int
            Number of most similar items to keep per liked anime.

        Returns
        -------
        np.ndarray or None
            1D CB score vector over items, or None if no liked anime.
        """
        # Filter ratings of the given user
        user_ratings = self.ratings_df[self.ratings_df["user_id"] == user_id]

        # Select anime rated above like_th
        liked_anime = user_ratings[user_ratings["rating"] >= like_th]["anime_id"].values

        # Convert liked MAL_IDs to indices in similarity matrix
        liked_idx = [self.malid_to_idx[aid] for aid in liked_anime if aid in self.malid_to_idx]

        # If user has no liked anime in our processed set, return None
        if not liked_idx:
            return None

        # Extract similarity rows for liked items
        liked_sims = self.item_sim_matrix[liked_idx]

        # Create boolean mask for top_k neighbors of each liked item
        mask = np.zeros_like(liked_sims, dtype=bool)
        for i in range(len(liked_idx)):
            # Indices of top_k similar items (descending order)
            top_idx = np.argsort(-liked_sims[i])[:top_k]
            mask[i, top_idx] = True

        # Zero out non-top_k similarities
        filtered = np.where(mask, liked_sims, 0.0)

        # Avoid division by zero: count how many times each item is selected
        denom = np.maximum(mask.sum(axis=0), 1e-8)

        # Aggregate similarities across liked items (mean of selected similarities)
        cb_vector = filtered.sum(axis=0) / denom

        return cb_vector

    # =====================================================
    # USER-BASED RECOMMENDATIONS (HYBRID: SVD + CB + META)
    # =====================================================
    def get_user_recommendations(self):
        """
        Generate personalized hybrid recommendations for a given user.

        Steps:
        1. Parse user_id and top_n from inputs.
        2. Check if user exists in ratings.
        3. Compute the CB preference vector using liked items.
        4. For each candidate anime (not yet watched):
           - Predict CF score using SVD.
           - Get CB score from CB vector.
           - Feed [CF, CB] into meta_model to get final score.
        5. Sort by final score and display top_n in the Treeview.
        """
        try:
            # Read and validate inputs
            user_id = int(self.user_entry.get())
            top_n = int(self.top_n_user.get())

            self.log("\n" + "=" * 50)
            self.log(f"Generating recommendations for User {user_id}")
            self.log("=" * 50)

            # Clear previous results in the Treeview
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Check whether the user exists in rating data
            user_history = self.ratings_df[self.ratings_df["user_id"] == user_id]
            if user_history.empty:
                messagebox.showwarning("Not Found", f"User {user_id} not found in rating data.")
                return

            watched_ids = set(user_history["anime_id"].values)
            self.log(f"User has rated {len(watched_ids)} anime.")

            # Compute CB scores for the user
            cb_vector = self.get_cb_scores_for_user(user_id)
            if cb_vector is None:
                messagebox.showwarning(
                    "No Liked Anime",
                    f"User {user_id} has no anime with rating >= 8.0 in this dataset."
                )
                return

            # Candidate items: all anime that user has not watched and that exist in mapping
            all_anime = self.anime_df["MAL_ID"].values
            candidates = [
                aid for aid in all_anime
                if aid not in watched_ids and aid in self.malid_to_idx
            ]

            self.log(f"Evaluating {len(candidates)} candidate anime...")

            scores = []
            # Compute hybrid scores for each candidate anime
            for anime_id in candidates:
                try:
                    # 1) Collaborative Filtering score via SVD
                    cf_score = self.svd_model.predict(user_id, anime_id).est

                    # 2) Content-based similarity score from CB vector
                    cb_score = cb_vector[self.malid_to_idx[anime_id]]

                    # 3) Meta-model prediction (fusion of CF + CB)
                    #    Input shape: [[cf_score, cb_score]]
                    meta_pred = self.meta_model.predict([[cf_score, cb_score]])[0]

                    # Clip final score to rating scale [1, 10]
                    meta_pred = float(np.clip(meta_pred, 1.0, 10.0))

                    scores.append((anime_id, meta_pred, cf_score, cb_score))
                except Exception:
                    # Skip any problematic candidate (e.g., missing mapping)
                    continue

            # Sort candidates by meta-model predicted score in descending order
            scores.sort(key=lambda x: x[1], reverse=True)
            top_recommendations = scores[:top_n]

            # Display results in the Treeview
            for rank, (anime_id, score, cf, cb) in enumerate(top_recommendations, start=1):
                anime_info = self.anime_df[self.anime_df["MAL_ID"] == anime_id].iloc[0]
                title = anime_info["Name"]

                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        rank,
                        title[:50],                 # Truncate title for neat display
                        f"{score:.2f}",             # Final predicted score
                        f"CF:{cf:.2f} | CB:{cb:.2f}"  # Show both components for transparency
                    )
                )

            self.log(f"Displayed top {top_n} recommendations.")
            self.log("=" * 50 + "\n")

        except ValueError:
            # Non-integer user_id or top_n
            messagebox.showerror("Invalid Input", "Please enter a valid integer User ID and Top N.")
        except Exception as e:
            # Any unexpected error during recommendation
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", f"An error occurred during recommendation:\n{str(e)}")

    # =====================================================
    # SIMILAR ANIME RECOMMENDATIONS (CONTENT-BASED)
    # =====================================================
    def get_similar_anime(self):
        """
        Find anime that are content-wise similar to a given title.

        Steps:
        1. Search anime_df for a title containing the input text.
        2. Use the first matched anime as the target.
        3. Look up its index in the similarity matrix.
        4. Sort items by similarity score and display the top N (excluding itself).
        """
        try:
            target_title = self.anime_entry.get().strip()
            top_n = int(self.top_n_anime.get())

            if not target_title:
                messagebox.showwarning("Empty Input", "Please enter an anime title.")
                return

            self.log("\n" + "=" * 50)
            self.log(f"Finding anime similar to '{target_title}'")
            self.log("=" * 50)

            # Clear previous results
            for item in self.tree.get_children():
                self.tree.delete(item)

            # Find anime with titles containing the query (case-insensitive)
            match = self.anime_df[
                self.anime_df["Name"].str.contains(target_title, case=False, na=False)
            ]

            if match.empty:
                messagebox.showwarning("Not Found", f"Anime '{target_title}' not found in anime.csv.")
                self.log("No title match found. Try a different keyword or spelling.")
                return

            # Use the first matching anime as the target
            target_id = match.iloc[0]["MAL_ID"]
            target_name = match.iloc[0]["Name"]
            self.log(f"Matched title: {target_name} (MAL_ID: {target_id})")

            # Check whether this anime is present in the processed similarity matrix mapping
            if target_id not in self.malid_to_idx:
                messagebox.showwarning(
                    "Not Available",
                    f"'{target_name}' is not included in the processed similarity dataset."
                )
                return

            # Retrieve similarity vector for target anime
            target_idx = self.malid_to_idx[target_id]
            similarities = self.item_sim_matrix[target_idx]

            # Get indices of top N similar items (excluding the target itself at index 0)
            top_similar_idx = np.argsort(-similarities)[1: top_n + 1]

            # Display similar anime
            for rank, idx in enumerate(top_similar_idx, start=1):
                anime_id = self.idx_to_malid[idx]
                anime_info = self.anime_df[self.anime_df["MAL_ID"] == anime_id].iloc[0]
                title = anime_info["Name"]
                sim_score = float(similarities[idx])

                self.tree.insert(
                    "",
                    tk.END,
                    values=(
                        rank,
                        title[:50],
                        f"{sim_score:.3f}",    # Use similarity as "score"
                        "Content Similarity"   # Explanation of the score type
                    )
                )

            self.log(f"Displayed top {top_n} similar anime.")
            self.log("=" * 50 + "\n")

        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for Top N.")
        except Exception as e:
            self.log(f"ERROR: {str(e)}")
            messagebox.showerror("Error", f"An error occurred while finding similar anime:\n{str(e)}")


# =====================================================
# MAIN ENTRY POINT
# =====================================================
def main():
    """
    Create the Tkinter root window and start the GUI application.
    """
    root = tk.Tk()
    app = AnimeRecommenderGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()


