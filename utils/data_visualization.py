                        flat_stat["good_count"] = settings.get("good_count", 0)
                        flat_stat["bad_count"] = settings.get("bad_count", 0)
                        flat_stat["threshold"] = settings.get("threshold", 0)
                
                flattened_stats.append(flat_stat)
            
            # Convert to DataFrame
            df = pd.DataFrame(flattened_stats)
            
            # Convert timestamp to datetime
            if "timestamp" in df.columns:
                df["timestamp"] = pd.to_datetime(df["timestamp"])
            
            return df
        except Exception as e:
            logger.error(f"Error loading system stats: {e}")
            raise VisualizationError(f"Failed to load system stats: {e}")
    
    def _save_figure(self, fig, filename, format="png", dpi=100):
        """
        Save a matplotlib figure to a file.
        
        Args:
            fig: Matplotlib figure
            filename: Output filename (without extension)
            format: Output format (png, jpg, pdf, svg)
            dpi: Resolution for raster formats
            
        Returns:
            Path to the saved file
        """
        if not HAS_MATPLOTLIB:
            raise VisualizationError("matplotlib is required for saving figures")
        
        # Create full path with extension
        output_path = os.path.join(self.output_dir, f"{filename}.{format}")
        
        try:
            # Save the figure
            fig.savefig(output_path, format=format, dpi=dpi, bbox_inches="tight")
            plt.close(fig)  # Close to free memory
            logger.debug(f"Saved visualization to {output_path}")
            return output_path
        except Exception as e:
            logger.error(f"Error saving figure: {e}")
            plt.close(fig)  # Close even on error
            raise VisualizationError(f"Failed to save figure: {e}")
    
    def plot_defect_history(self, csv_path=None, days=30, output_file="defect_history", format="png"):
        """
        Generate a plot showing defect detection history over time.
        
        Args:
            csv_path: Path to the CSV file with detection results
            days: Number of days to include
            output_file: Output filename (without extension)
            format: Output format (png, jpg, pdf, svg)
            
        Returns:
            Path to the saved visualization
        """
        if not HAS_MATPLOTLIB or not HAS_PANDAS:
            raise VisualizationError("matplotlib and pandas are required for plotting")
        
        # Default CSV path
        if csv_path is None:
            csv_path = config.PATHS.get("CSV_FILENAME")
        
        try:
            # Load data
            df = self._load_csv_data(csv_path)
            
            # Parse date from filename if not already a column
            if "date" not in df.columns and "filename" in df.columns:
                # Extract date from filenames like image_20230131_120000.jpg
                df["date"] = df["filename"].str.extract(r"(\d{8})").astype(str)
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            
            # If no date column, create one based on the file modification time
            if "date" not in df.columns:
                df["date"] = datetime.datetime.now().date()
            
            # Filter for the requested time period
            if isinstance(df["date"].iloc[0], pd.Timestamp):
                start_date = datetime.datetime.now() - datetime.timedelta(days=days)
                df = df[df["date"] >= start_date]
            
            # Group by date and calculate defect rate
            daily_stats = df.groupby(df["date"].dt.date).agg(
                total=("is_defect", "count"),
                defects=("is_defect", lambda x: (x == "Yes").sum() if x.dtype == "object" else x.sum())
            ).reset_index()
            
            daily_stats["defect_rate"] = daily_stats["defects"] / daily_stats["total"] * 100
            
            # Create figure
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot defect rate
            ax1.plot(daily_stats["date"], daily_stats["defect_rate"], "r-", linewidth=2)
            ax1.set_ylabel("Defect Rate (%)", color="r")
            ax1.tick_params(axis="y", labelcolor="r")
            ax1.set_ylim(bottom=0)
            
            # Create second y-axis for counts
            ax2 = ax1.twinx()
            
            # Plot total inspections and defects
            ax2.bar(daily_stats["date"], daily_stats["total"], alpha=0.3, color="b", label="Total Inspections")
            ax2.bar(daily_stats["date"], daily_stats["defects"], alpha=0.5, color="r", label="Defects")
            ax2.set_ylabel("Count", color="b")
            ax2.tick_params(axis="y", labelcolor="b")
            ax2.legend(loc="upper right")
            
            # Set x-axis format
            ax1.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            if len(daily_stats) > 14:
                ax1.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(daily_stats) // 14)))
            fig.autofmt_xdate()
            
            # Set title and labels
            ax1.set_title(f"Defect Detection History (Last {days} Days)")
            ax1.set_xlabel("Date")
            
            # Add grid
            ax1.grid(True, alpha=0.3)
            
            # Save the figure
            return self._save_figure(fig, output_file, format)
        except Exception as e:
            logger.error(f"Error plotting defect history: {e}")
            raise VisualizationError(f"Failed to plot defect history: {e}")
    
    def plot_error_distribution(self, csv_path=None, bins=50, output_file="error_distribution", format="png"):
        """
        Plot the distribution of reconstruction errors with threshold indicators.
        
        Args:
            csv_path: Path to the CSV file with detection results
            bins: Number of bins for the histogram
            output_file: Output filename (without extension)
            format: Output format (png, jpg, pdf, svg)
            
        Returns:
            Path to the saved visualization
        """
        if not HAS_MATPLOTLIB or not HAS_PANDAS:
            raise VisualizationError("matplotlib and pandas are required for plotting")
        
        # Default CSV path
        if csv_path is None:
            csv_path = config.PATHS.get("CSV_FILENAME")
        
        try:
            # Load data
            df = self._load_csv_data(csv_path)
            
            # Check for required columns
            if "global_error" not in df.columns:
                raise VisualizationError("CSV file must contain 'global_error' column")
            
            # Check if we have the threshold column
            has_threshold = "global_threshold" in df.columns
            
            # Create figure
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Determine good and bad samples
            if "is_defect" in df.columns:
                if df["is_defect"].dtype == "object":
                    good_mask = df["is_defect"] == "No"
                    bad_mask = df["is_defect"] == "Yes"
                else:
                    good_mask = df["is_defect"] == 0
                    bad_mask = df["is_defect"] == 1
                
                # Plot histograms for good and bad samples
                ax.hist(df.loc[good_mask, "global_error"], bins=bins, alpha=0.5, color="g", label="Normal")
                ax.hist(df.loc[bad_mask, "global_error"], bins=bins, alpha=0.5, color="r", label="Defect")
            else:
                # Single histogram if no classification
                ax.hist(df["global_error"], bins=bins, alpha=0.7, color="b")
            
            # Add threshold line if available
            if has_threshold:
                threshold = df["global_threshold"].iloc[-1]  # Use the most recent threshold
                ax.axvline(x=threshold, color="y", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.3f})")
            
            # Set title and labels
            ax.set_title("Distribution of Reconstruction Errors")
            ax.set_xlabel("Reconstruction Error")
            ax.set_ylabel("Frequency")
            ax.legend()
            
            # Add grid
            ax.grid(True, alpha=0.3)
            
            # Save the figure
            return self._save_figure(fig, output_file, format)
        except Exception as e:
            logger.error(f"Error plotting error distribution: {e}")
            raise VisualizationError(f"Failed to plot error distribution: {e}")
    
    def plot_system_metrics(self, stats_path=None, days=7, output_file="system_metrics", format="png"):
        """
        Plot system metrics (CPU, memory, disk) over time.
        
        Args:
            stats_path: Path to the system stats JSON file
            days: Number of days to include
            output_file: Output filename (without extension)
            format: Output format (png, jpg, pdf, svg)
            
        Returns:
            Path to the saved visualization
        """
        if not HAS_MATPLOTLIB or not HAS_PANDAS:
            raise VisualizationError("matplotlib and pandas are required for plotting")
        
        try:
            # Load system stats
            df = self._load_system_stats(stats_path)
            
            # Filter for the requested time period
            if not df.empty and "timestamp" in df.columns:
                start_time = datetime.datetime.now() - datetime.timedelta(days=days)
                df = df[df["timestamp"] >= start_time]
            
            if df.empty:
                raise VisualizationError("No system stats data available")
            
            # Create figure with subplots
            fig, axs = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
            
            # Plot CPU usage
            if "cpu_usage" in df.columns:
                axs[0].plot(df["timestamp"], df["cpu_usage"], "r-", linewidth=2)
                axs[0].set_title("CPU Usage")
                axs[0].set_ylabel("Usage (%)")
                axs[0].set_ylim(0, 100)
                axs[0].grid(True, alpha=0.3)
            
            # Plot memory usage
            if "memory_usage_percent" in df.columns:
                axs[1].plot(df["timestamp"], df["memory_usage_percent"], "g-", linewidth=2)
                axs[1].set_title("Memory Usage")
                axs[1].set_ylabel("Usage (%)")
                axs[1].set_ylim(0, 100)
                axs[1].grid(True, alpha=0.3)
            
            # Plot disk usage
            if "disk_usage_percent" in df.columns:
                axs[2].plot(df["timestamp"], df["disk_usage_percent"], "b-", linewidth=2)
                axs[2].set_title("Disk Usage")
                axs[2].set_ylabel("Usage (%)")
                axs[2].set_ylim(0, 100)
                axs[2].grid(True, alpha=0.3)
            
            # Format x-axis
            axs[2].set_xlabel("Time")
            for ax in axs:
                ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
            
            # Add overall title
            fig.suptitle(f"System Metrics (Last {days} Days)", fontsize=16)
            
            # Adjust layout
            fig.tight_layout(rect=[0, 0, 1, 0.97])
            
            # Save the figure
            return self._save_figure(fig, output_file, format)
        except Exception as e:
            logger.error(f"Error plotting system metrics: {e}")
            raise VisualizationError(f"Failed to plot system metrics: {e}")
    
    def plot_heatmap(self, image_path, model=None, output_file=None, format="png"):
        """
        Generate a heatmap visualization for an image, showing areas of high reconstruction error.
        
        Args:
            image_path: Path to the input image
            model: Model to use for reconstruction (if None, uses active model)
            output_file: Output filename (without extension)
            format: Output format (png, jpg, pdf, svg)
            
        Returns:
            Path to the saved visualization
        """
        if not HAS_MATPLOTLIB:
            raise VisualizationError("matplotlib is required for heatmap generation")
        
        # Try to import PyTorch for model inference
        try:
            import torch
            import torch.nn as nn
            import torchvision.transforms as transforms
            from PIL import Image
        except ImportError:
            raise VisualizationError("PyTorch is required for heatmap generation")
        
        try:
            # Default output file based on input image name
            if output_file is None:
                base_name = os.path.splitext(os.path.basename(image_path))[0]
                output_file = f"{base_name}_heatmap"
            
            # Load the model if not provided
            if model is None:
                # Import model manager
                from models.model_manager import ModelManager
                model_manager = ModelManager()
                model = model_manager.load_model()
            
            # Load and preprocess the image
            transform = transforms.Compose([
                transforms.Resize(config.MODEL["IMAGE_SIZE"]),
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=config.MODEL["NORMALIZE_MEAN"],
                    std=config.MODEL["NORMALIZE_STD"]
                )
            ])
            
            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0)
            
            # Get reconstruction
            model.eval()
            with torch.no_grad():
                reconstructed, _ = model(img_tensor)
            
            # Calculate pixel-wise error
            criterion = nn.MSELoss(reduction='none')
            error = criterion(reconstructed, img_tensor)
            error = error.mean(dim=1).squeeze().cpu().numpy()
            
            # Normalize error map
            error_min, error_max = error.min(), error.max()
            if error_max > error_min:
                error_norm = (error - error_min) / (error_max - error_min)
            else:
                error_norm = error
            
            # Create figure
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
            
            # Display original image
            img_np = np.array(img.resize(config.MODEL["IMAGE_SIZE"]))
            ax1.imshow(img_np)
            ax1.set_title("Original Image")
            ax1.axis("off")
            
            # Display heatmap
            cmap = plt.cm.jet
            cmap.set_bad(alpha=0)
            im = ax2.imshow(error_norm, cmap=cmap, vmin=0, vmax=1)
            ax2.set_title("Error Heatmap")
            ax2.axis("off")
            
            # Add colorbar
            cbar = fig.colorbar(im, ax=ax2, orientation="vertical", fraction=0.046, pad=0.04)
            cbar.set_label("Reconstruction Error")
            
            # Add overall title
            fig.suptitle("Defect Analysis Heatmap", fontsize=16)
            
            # Save the figure
            return self._save_figure(fig, output_file, format)
        except Exception as e:
            logger.error(f"Error generating heatmap: {e}")
            raise VisualizationError(f"Failed to generate heatmap: {e}")
    
    def create_dashboard(self, output_file="dashboard", format="png"):
        """
        Create a comprehensive dashboard with multiple visualizations.
        
        Args:
            output_file: Output filename (without extension)
            format: Output format (png, jpg, pdf, svg)
            
        Returns:
            Path to the saved dashboard
        """
        if not HAS_MATPLOTLIB or not HAS_PANDAS:
            raise VisualizationError("matplotlib and pandas are required for dashboard creation")
        
        try:
            # Create a large figure
            fig = plt.figure(figsize=(20, 15))
            
            # Define grid layout
            gs = plt.GridSpec(3, 2, figure=fig, wspace=0.3, hspace=0.4)
            
            # Add title
            fig.suptitle("Defect Detection System Dashboard", fontsize=24, y=0.98)
            
            # Get current date and time
            now = datetime.datetime.now()
            timestamp = now.strftime("%Y-%m-%d %H:%M:%S")
            
            # Add timestamp
            fig.text(0.5, 0.94, f"Generated: {timestamp}", fontsize=12, ha="center")
            
            # Plot defect history
            ax1 = fig.add_subplot(gs[0, :])
            self._plot_defect_history_on_axis(ax1)
            
            # Plot error distribution
            ax2 = fig.add_subplot(gs[1, 0])
            self._plot_error_distribution_on_axis(ax2)
            
            # Plot system metrics
            ax3 = fig.add_subplot(gs[1, 1])
            self._plot_system_metrics_on_axis(ax3)
            
            # Plot detection counts
            ax4 = fig.add_subplot(gs[2, 0])
            self._plot_detection_counts_on_axis(ax4)
            
            # Plot threshold history
            ax5 = fig.add_subplot(gs[2, 1])
            self._plot_threshold_history_on_axis(ax5)
            
            # Save the figure
            return self._save_figure(fig, output_file, format, dpi=150)
        except Exception as e:
            logger.error(f"Error creating dashboard: {e}")
            raise VisualizationError(f"Failed to create dashboard: {e}")
    
    def _plot_defect_history_on_axis(self, ax):
        """Plot defect history on a given axis."""
        try:
            # Load data
            csv_path = config.PATHS.get("CSV_FILENAME")
            df = self._load_csv_data(csv_path)
            
            # Parse date from filename if not already a column
            if "date" not in df.columns and "filename" in df.columns:
                # Extract date from filenames like image_20230131_120000.jpg
                df["date"] = df["filename"].str.extract(r"(\d{8})").astype(str)
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            
            # If no date column, create one based on the file modification time
            if "date" not in df.columns:
                df["date"] = datetime.datetime.now().date()
            
            # Filter for the last 30 days
            start_date = datetime.datetime.now() - datetime.timedelta(days=30)
            df = df[df["date"] >= start_date]
            
            # Group by date and calculate defect rate
            daily_stats = df.groupby(df["date"].dt.date).agg(
                total=("is_defect", "count"),
                defects=("is_defect", lambda x: (x == "Yes").sum() if x.dtype == "object" else x.sum())
            ).reset_index()
            
            daily_stats["defect_rate"] = daily_stats["defects"] / daily_stats["total"] * 100
            
            # Plot defect rate
            ax.plot(daily_stats["date"], daily_stats["defect_rate"], "r-", linewidth=2)
            ax.set_ylabel("Defect Rate (%)", color="r")
            ax.tick_params(axis="y", labelcolor="r")
            ax.set_ylim(bottom=0)
            
            # Create second y-axis for counts
            ax2 = ax.twinx()
            
            # Plot total inspections and defects
            ax2.bar(daily_stats["date"], daily_stats["total"], alpha=0.3, color="b", label="Total Inspections")
            ax2.bar(daily_stats["date"], daily_stats["defects"], alpha=0.5, color="r", label="Defects")
            ax2.set_ylabel("Count", color="b")
            ax2.tick_params(axis="y", labelcolor="b")
            ax2.legend(loc="upper right")
            
            # Set x-axis format
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            if len(daily_stats) > 14:
                ax.xaxis.set_major_locator(mdates.DayLocator(interval=max(1, len(daily_stats) // 14)))
            
            # Set title and labels
            ax.set_title("Defect Detection History (Last 30 Days)")
            ax.set_xlabel("Date")
            
            # Add grid
            ax.grid(True, alpha=0.3)
        except Exception as e:
            logger.error(f"Error plotting defect history on axis: {e}")
            ax.text(0.5, 0.5, "Error loading defect history data", 
                   ha="center", va="center", transform=ax.transAxes)
    
    def _plot_error_distribution_on_axis(self, ax):
        """Plot error distribution on a given axis."""
        try:
            # Load data
            csv_path = config.PATHS.get("CSV_FILENAME")
            df = self._load_csv_data(csv_path)
            
            # Check for required columns
            if "global_error" not in df.columns:
                raise VisualizationError("CSV file must contain 'global_error' column")
            
            # Check if we have the threshold column
            has_threshold = "global_threshold" in df.columns
            
            # Determine good and bad samples
            if "is_defect" in df.columns:
                if df["is_defect"].dtype == "object":
                    good_mask = df["is_defect"] == "No"
                    bad_mask = df["is_defect"] == "Yes"
                else:
                    good_mask = df["is_defect"] == 0
                    bad_mask = df["is_defect"] == 1
                
                # Plot histograms for good and bad samples
                ax.hist(df.loc[good_mask, "global_error"], bins=30, alpha=0.5, color="g", label="Normal")
                ax.hist(df.loc[bad_mask, "global_error"], bins=30, alpha=0.5, color="r", label="Defect")
            else:
                # Single histogram if no classification
                ax.hist(df["global_error"], bins=30, alpha=0.7, color="b")
            
            # Add threshold line if available
            if has_threshold:
                threshold = df["global_threshold"].iloc[-1]  # Use the most recent threshold
                ax.axvline(x=threshold, color="y", linestyle="--", linewidth=2, label=f"Threshold ({threshold:.3f})")
            
            # Set title and labels
            ax.set_title("Distribution of Reconstruction Errors")
            ax.set_xlabel("Reconstruction Error")
            ax.set_ylabel("Frequency")
            ax.legend()
            
            # Add grid
            ax.grid(True, alpha=0.3)
        except Exception as e:
            logger.error(f"Error plotting error distribution on axis: {e}")
            ax.text(0.5, 0.5, "Error loading error distribution data", 
                   ha="center", va="center", transform=ax.transAxes)
    
    def _plot_system_metrics_on_axis(self, ax):
        """Plot system metrics on a given axis."""
        try:
            # Load system stats
            df = self._load_system_stats()
            
            # Filter for the last 7 days
            if not df.empty and "timestamp" in df.columns:
                start_time = datetime.datetime.now() - datetime.timedelta(days=7)
                df = df[df["timestamp"] >= start_time]
            
            if df.empty:
                raise VisualizationError("No system stats data available")
            
            # Create metrics to plot
            metrics = []
            if "cpu_usage" in df.columns:
                metrics.append(("cpu_usage", "CPU", "r"))
            if "memory_usage_percent" in df.columns:
                metrics.append(("memory_usage_percent", "Memory", "g"))
            if "disk_usage_percent" in df.columns:
                metrics.append(("disk_usage_percent", "Disk", "b"))
            
            # Plot each metric
            for col, label, color in metrics:
                ax.plot(df["timestamp"], df[col], color=color, linewidth=2, label=f"{label}")
            
            # Set title and labels
            ax.set_title("System Resource Usage")
            ax.set_xlabel("Time")
            ax.set_ylabel("Usage (%)")
            ax.set_ylim(0, 100)
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            
            # Add legend and grid
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            logger.error(f"Error plotting system metrics on axis: {e}")
            ax.text(0.5, 0.5, "Error loading system metrics data", 
                   ha="center", va="center", transform=ax.transAxes)
    
    def _plot_detection_counts_on_axis(self, ax):
        """Plot detection counts on a given axis."""
        try:
            # Get settings
            settings_file = config.PATHS.get("SETTINGS_FILE")
            if settings_file and os.path.exists(settings_file):
                with open(settings_file, "r") as f:
                    settings = json.load(f)
                
                # Extract counts
                good_count = settings.get("good_count", 0)
                bad_count = settings.get("bad_count", 0)
                
                # Plot counts
                labels = ["Normal", "Defective"]
                counts = [good_count, bad_count]
                colors = ["g", "r"]
                
                ax.bar(labels, counts, color=colors, alpha=0.7)
                
                # Add count labels on bars
                for i, count in enumerate(counts):
                    ax.text(i, count + (max(counts) * 0.02), str(count), 
                           ha="center", va="bottom", fontsize=12)
                
                # Set title and labels
                ax.set_title("Detection Counts")
                ax.set_ylabel("Count")
                
                # Add percentage text
                total = sum(counts)
                if total > 0:
                    good_pct = good_count / total * 100
                    bad_pct = bad_count / total * 100
                    
                    # Add text box with percentages
                    textstr = f"Normal: {good_pct:.1f}%\nDefective: {bad_pct:.1f}%"
                    props = dict(boxstyle="round", facecolor="wheat", alpha=0.5)
                    ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=10,
                           verticalalignment="top", bbox=props)
                
                # Add grid
                ax.grid(True, alpha=0.3)
            else:
                ax.text(0.5, 0.5, "Settings file not found", 
                       ha="center", va="center", transform=ax.transAxes)
        except Exception as e:
            logger.error(f"Error plotting detection counts on axis: {e}")
            ax.text(0.5, 0.5, "Error loading detection count data", 
                   ha="center", va="center", transform=ax.transAxes)
    
    def _plot_threshold_history_on_axis(self, ax):
        """Plot threshold history on a given axis."""
        try:
            # Load data
            csv_path = config.PATHS.get("CSV_FILENAME")
            df = self._load_csv_data(csv_path)
            
            # Check for required columns
            if "global_threshold" not in df.columns:
                raise VisualizationError("CSV file must contain 'global_threshold' column")
            
            # Parse date from filename if not already a column
            if "date" not in df.columns and "filename" in df.columns:
                # Extract date from filenames like image_20230131_120000.jpg
                df["date"] = df["filename"].str.extract(r"(\d{8})").astype(str)
                df["date"] = pd.to_datetime(df["date"], format="%Y%m%d", errors="coerce")
            
            # If no date column, create a dummy date column
            if "date" not in df.columns:
                df["date"] = pd.date_range(end=datetime.datetime.now(), periods=len(df))
            
            # Sort by date
            df = df.sort_values("date")
            
            # Plot threshold history
            ax.plot(df["date"], df["global_threshold"], "y-", linewidth=2, label="Global Threshold")
            
            # Plot patch threshold if available
            if "patch_threshold" in df.columns:
                ax.plot(df["date"], df["patch_threshold"], "g--", linewidth=2, label="Patch Threshold")
            
            # Plot patch defect ratio if available
            if "patch_defect_ratio_threshold" in df.columns:
                ax.plot(df["date"], df["patch_defect_ratio_threshold"], "b:", linewidth=2, label="Defect Ratio")
            
            # Set title and labels
            ax.set_title("Threshold History")
            ax.set_xlabel("Date")
            ax.set_ylabel("Threshold Value")
            
            # Format x-axis
            ax.xaxis.set_major_formatter(mdates.DateFormatter("%m-%d"))
            
            # Add legend and grid
            ax.legend()
            ax.grid(True, alpha=0.3)
        except Exception as e:
            logger.error(f"Error plotting threshold history on axis: {e}")
            ax.text(0.5, 0.5, "Error loading threshold history data", 
                   ha="center", va="center", transform=ax.transAxes)

# For testing
if __name__ == "__main__":
    print("Testing Data Visualizer")
    
    # Check if matplotlib is available
    if not HAS_MATPLOTLIB:
        print("WARNING: matplotlib not available. Skipping tests.")
        import sys
        sys.exit(0)
    
    # Create visualizer
    visualizer = DataVisualizer()
    
    # Test defect history plot
    try:
        print("\nTesting defect history plot:")
        csv_path = config.PATHS.get("CSV_FILENAME")
        if os.path.exists(csv_path):
            output_path = visualizer.plot_defect_history(csv_path, days=30)
            print(f"Generated defect history plot: {output_path}")
        else:
            print(f"CSV file not found: {csv_path}")
    except Exception as e:
        print(f"Error generating defect history plot: {e}")
    
    # Test system metrics plot
    try:
        print("\nTesting system metrics plot:")
        stats_path = os.path.join(config.BASE_DIR, "system_stats.json")
        if os.path.exists(stats_path):
            output_path = visualizer.plot_system_metrics(stats_path, days=7)
            print(f"Generated system metrics plot: {output_path}")
        else:
            print(f"System stats file not found: {stats_path}")
    except Exception as e:
        print(f"Error generating system metrics plot: {e}")
    
    # Test dashboard creation
    try:
        print("\nTesting dashboard creation:")
        output_path = visualizer.create_dashboard()
        print(f"Generated dashboard: {output_path}")
    except Exception as e:
        print(f"Error generating dashboard: {e}")
    
    print("\nData Visualizer tests completed.")#!/usr/bin/env python3
"""
Data Visualization Utilities for Defect Detection System
Generates charts, plots, and visualizations of detection results and system performance.
"""

import os
import json
import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Union

# Import centralized configuration
from config import config

# Set up logger
logger = config.setup_logger("data_visualization")

# Try to import visualization libraries
try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend for headless operation
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.figure import Figure
    import numpy as np
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    logger.warning("matplotlib not available. Visualization functionality will be limited.")

# Try to import pandas for data manipulation
try:
    import pandas as pd
    HAS_PANDAS = True
except ImportError:
    HAS_PANDAS = False
    logger.warning("pandas not available. Some data processing features will be limited.")

class VisualizationError(Exception):
    """Custom exception for visualization errors"""
    pass

class DataVisualizer:
    """
    Generates charts and visualizations for defect detection results and system metrics.
    """
    
    def __init__(self, output_dir=None):
        """
        Initialize the data visualizer.
        
        Args:
            output_dir: Directory for saving visualization outputs
        """
        self.output_dir = output_dir or os.path.join(config.BASE_DIR, "visualizations")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Check for required libraries
        if not HAS_MATPLOTLIB:
            logger.error("matplotlib is required for visualization")
        
        # Set default style
        if HAS_MATPLOTLIB:
            plt.style.use('dark_background')
            
            # Set default figure size and DPI
            plt.rcParams['figure.figsize'] = (12, 8)
            plt.rcParams['figure.dpi'] = 100
            plt.rcParams['font.size'] = 12
    
    def _load_csv_data(self, csv_path):
        """
        Load data from CSV file.
        
        Args:
            csv_path: Path to the CSV file
            
        Returns:
            DataFrame with data or None if failed
        """
        if not HAS_PANDAS:
            raise VisualizationError("pandas is required for CSV data loading")
        
        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found: {csv_path}")
        
        try:
            # Load CSV with appropriate data types
            df = pd.read_csv(csv_path)
            return df
        except Exception as e:
            logger.error(f"Error loading CSV data: {e}")
            raise VisualizationError(f"Failed to load CSV data: {e}")
    
    def _load_system_stats(self, stats_path=None):
        """
        Load system statistics from JSON file.
        
        Args:
            stats_path: Path to the stats JSON file
            
        Returns:
            DataFrame with stats data or None if failed
        """
        if not HAS_PANDAS:
            raise VisualizationError("pandas is required for stats data loading")
        
        # Default stats path
        if stats_path is None:
            stats_path = os.path.join(config.BASE_DIR, "system_stats.json")
        
        if not os.path.exists(stats_path):
            raise FileNotFoundError(f"Stats file not found: {stats_path}")
        
        try:
            # Load stats from JSON
            with open(stats_path, "r") as f:
                stats_data = json.load(f)
            
            # Convert to DataFrame
            if not stats_data:
                return pd.DataFrame()
            
            # Normalize the JSON data (flatten nested structures)
            flattened_stats = []
            for stat in stats_data:
                flat_stat = {"timestamp": stat.get("timestamp")}
                
                # Flatten disk usage
                if "disk_usage" in stat and stat["disk_usage"] is not None:
                    flat_stat["disk_usage_percent"] = stat["disk_usage"].get("percent")
                
                # Flatten memory usage
                if "memory_usage" in stat and stat["memory_usage"] is not None:
                    flat_stat["memory_usage_percent"] = stat["memory_usage"].get("percent")
                
                # Add other metrics
                flat_stat["cpu_usage"] = stat.get("cpu_usage")
                flat_stat["temperature"] = stat.get("temperature")
                
                # Add inference stats
                if "inference_stats" in stat and stat["inference_stats"] is not None:
                    if "images" in stat["inference_stats"]:
                        flat_stat["normal_images"] = stat["inference_stats"]["images"].get("normal_count", 0)
                        flat_stat["anomaly_images"] = stat["inference_stats"]["images"].get("anomaly_count", 0)
                    
                    if "settings" in stat["inference_stats"]:
                        settings = stat["inference_stats"]["settings"]
                        flat_stat["good_count"] = settings.get("good_count", 0)
                        flat_stat["bad_count"] = settings.get("bad_count", 