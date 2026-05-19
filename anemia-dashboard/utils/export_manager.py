"""Export manager for converting dashboard visualizations to PNG format for research papers."""
from __future__ import annotations

import io
import logging
import tempfile
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import folium
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image, ImageDraw, ImageFont
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

logger = logging.getLogger(__name__)

# Export configuration
EXPORT_DPI = 300
EXPORT_WIDTH = 2400  # pixels at 300 DPI = 8 inches
EXPORT_HEIGHT = 1800  # pixels at 300 DPI = 6 inches
COMPOSITE_WIDTH = 2400
COMPOSITE_SPACING = 100  # pixels between charts in composite


def export_plotly_chart(
    fig: go.Figure,
    filename: str,
    width: int = EXPORT_WIDTH,
    height: int = EXPORT_HEIGHT,
    scale: float = 3.0,
) -> bytes:
    """Export Plotly figure to high-resolution PNG.

    Args:
        fig: Plotly figure object.
        filename: Target filename (for logging).
        width: Width in pixels.
        height: Height in pixels.
        scale: Scale factor for higher resolution (3.0 = ~300 DPI).

    Returns:
        PNG image bytes.
    """
    try:
        img_bytes = fig.to_image(
            format="png",
            width=width,
            height=height,
            scale=scale,
            engine="kaleido",
        )
        logger.info("Exported Plotly chart: %s", filename)
        return img_bytes
    except Exception as exc:
        logger.exception("Failed to export Plotly chart %s: %s", filename, exc)
        raise


def export_dataframe_table(
    df: pd.DataFrame,
    filename: str,
    title: Optional[str] = None,
    max_rows: int = 50,
    fig_width: int = 12,
    fig_height: Optional[int] = None,
) -> bytes:
    """Export DataFrame as styled PNG image.

    Args:
        df: DataFrame to export.
        filename: Target filename (for logging).
        title: Optional title to display above table.
        max_rows: Maximum rows to include (for large tables).
        fig_width: Figure width in inches.
        fig_height: Figure height in inches (auto if None).

    Returns:
        PNG image bytes.
    """
    try:
        # Limit rows for readability
        display_df = df.head(max_rows) if len(df) > max_rows else df.copy()
        
        # Calculate figure height based on rows
        if fig_height is None:
            fig_height = max(6, min(20, len(display_df) * 0.3 + 2))
        
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=EXPORT_DPI)
        ax.axis('tight')
        ax.axis('off')
        
        # Add title if provided
        if title:
            fig.suptitle(title, fontsize=16, fontweight='bold', y=0.98)
        
        # Create table
        table = ax.table(
            cellText=display_df.values,
            colLabels=display_df.columns,
            cellLoc='left',
            loc='center',
            bbox=[0, 0, 1, 1],
        )
        
        table.auto_set_font_size(False)
        table.set_fontsize(8)
        table.scale(1, 2)
        
        # Style header
        for key, cell in table.get_celld().items():
            if key[0] == 0:  # Header row
                cell.set_facecolor('#4472C4')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#F2F2F2' if key[0] % 2 == 0 else 'white')
        
        # Save to bytes
        buf = io.BytesIO()
        plt.savefig(buf, format='png', dpi=EXPORT_DPI, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        buf.seek(0)
        
        logger.info("Exported DataFrame table: %s", filename)
        return buf.getvalue()
    
    except Exception as exc:
        logger.exception("Failed to export DataFrame table %s: %s", filename, exc)
        raise


def export_folium_map(
    folium_map: folium.Map,
    filename: str,
    width: int = EXPORT_WIDTH,
    height: int = EXPORT_HEIGHT,
) -> bytes:
    """Export Folium map to PNG using Selenium.

    Args:
        folium_map: Folium map object.
        filename: Target filename (for logging).
        width: Width in pixels.
        height: Height in pixels.

    Returns:
        PNG image bytes.
    """
    try:
        # Save map to temporary HTML file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.html', delete=False) as tmp_file:
            tmp_path = Path(tmp_file.name)
            folium_map.save(str(tmp_path))
        
        # Setup Selenium WebDriver
        chrome_options = Options()
        chrome_options.add_argument('--headless')
        chrome_options.add_argument('--no-sandbox')
        chrome_options.add_argument('--disable-dev-shm-usage')
        chrome_options.add_argument(f'--window-size={width},{height}')
        
        service = Service(ChromeDriverManager().install())
        driver = webdriver.Chrome(service=service, options=chrome_options)
        
        try:
            # Load map and take screenshot
            driver.get(f'file://{tmp_path.absolute()}')
            driver.implicitly_wait(2)  # Wait for map to load
            
            screenshot_bytes = driver.get_screenshot_as_png()
            
            # Convert to high-res if needed
            img = Image.open(io.BytesIO(screenshot_bytes))
            
            # Ensure proper size
            if img.size != (width, height):
                img = img.resize((width, height), Image.Resampling.LANCZOS)
            
            buf = io.BytesIO()
            img.save(buf, format='PNG', dpi=(EXPORT_DPI, EXPORT_DPI))
            buf.seek(0)
            
            logger.info("Exported Folium map: %s", filename)
            return buf.getvalue()
        
        finally:
            driver.quit()
            tmp_path.unlink(missing_ok=True)
    
    except Exception as exc:
        logger.exception("Failed to export Folium map %s: %s", filename, exc)
        raise


def export_wordcloud_image(
    wordcloud_img: Image.Image,
    filename: str,
    width: int = EXPORT_WIDTH,
    height: int = EXPORT_HEIGHT,
) -> bytes:
    """Export WordCloud PIL image to high-resolution PNG.

    Args:
        wordcloud_img: PIL Image from WordCloud.
        filename: Target filename (for logging).
        width: Width in pixels.
        height: Height in pixels.

    Returns:
        PNG image bytes.
    """
    try:
        # Resize to target dimensions with high quality
        img_resized = wordcloud_img.resize((width, height), Image.Resampling.LANCZOS)
        
        buf = io.BytesIO()
        img_resized.save(buf, format='PNG', dpi=(EXPORT_DPI, EXPORT_DPI))
        buf.seek(0)
        
        logger.info("Exported WordCloud: %s", filename)
        return buf.getvalue()
    
    except Exception as exc:
        logger.exception("Failed to export WordCloud %s: %s", filename, exc)
        raise


def create_title_image(
    title: str,
    width: int = COMPOSITE_WIDTH,
    height: int = 150,
    font_size: int = 48,
) -> Image.Image:
    """Create a title image for composite layouts.

    Args:
        title: Title text.
        width: Image width in pixels.
        height: Image height in pixels.
        font_size: Font size for title.

    Returns:
        PIL Image with title.
    """
    img = Image.new('RGB', (width, height), color='white')
    draw = ImageDraw.Draw(img)
    
    try:
        # Try to use a nice font, fallback to default
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", font_size)
    except Exception:
        font = ImageFont.load_default()
    
    # Center the text
    bbox = draw.textbbox((0, 0), title, font=font)
    text_width = bbox[2] - bbox[0]
    text_height = bbox[3] - bbox[1]
    x = (width - text_width) // 2
    y = (height - text_height) // 2
    
    draw.text((x, y), title, fill='black', font=font)
    
    return img


def create_page_composite(
    images: List[Tuple[str, bytes]],
    page_title: str,
    output_width: int = COMPOSITE_WIDTH,
    spacing: int = COMPOSITE_SPACING,
) -> bytes:
    """Combine multiple images into a single composite image.

    Args:
        images: List of (title, image_bytes) tuples.
        page_title: Overall page title.
        output_width: Width of composite image.
        spacing: Vertical spacing between images.

    Returns:
        PNG image bytes of composite.
    """
    try:
        if not images:
            raise ValueError("No images provided for composite")
        
        # Load all images and calculate total height
        pil_images: List[Tuple[str, Image.Image]] = []
        total_height = 150  # Space for page title
        
        for title, img_bytes in images:
            img = Image.open(io.BytesIO(img_bytes))
            # Resize to standard width while maintaining aspect ratio
            aspect_ratio = img.height / img.width
            new_height = int(output_width * aspect_ratio)
            img_resized = img.resize((output_width, new_height), Image.Resampling.LANCZOS)
            pil_images.append((title, img_resized))
            total_height += 80  # Title space
            total_height += new_height
            total_height += spacing
        
        # Create composite canvas
        composite = Image.new('RGB', (output_width, total_height), color='white')
        
        # Add page title
        title_img = create_title_image(page_title, width=output_width, height=150)
        composite.paste(title_img, (0, 0))
        
        # Paste images
        y_offset = 150
        for title, img in pil_images:
            # Add subtitle
            subtitle_img = create_title_image(title, width=output_width, height=80, font_size=32)
            composite.paste(subtitle_img, (0, y_offset))
            y_offset += 80
            
            # Add image
            composite.paste(img, (0, y_offset))
            y_offset += img.height + spacing
        
        # Save composite
        buf = io.BytesIO()
        composite.save(buf, format='PNG', dpi=(EXPORT_DPI, EXPORT_DPI))
        buf.seek(0)
        
        logger.info("Created page composite: %s", page_title)
        return buf.getvalue()
    
    except Exception as exc:
        logger.exception("Failed to create composite for %s: %s", page_title, exc)
        raise


def create_export_zip(
    exports: Dict[str, Dict[str, bytes]],
    include_composites: bool = True,
) -> bytes:
    """Package all exports into a ZIP file.

    Args:
        exports: Nested dict {page_name: {filename: image_bytes}}.
        include_composites: Whether to create and include composite images.

    Returns:
        ZIP file bytes.
    """
    try:
        zip_buffer = io.BytesIO()
        
        with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
            for page_name, files in exports.items():
                # Add individual files
                for filename, img_bytes in files.items():
                    zip_path = f"{page_name}/{filename}"
                    zip_file.writestr(zip_path, img_bytes)
                
                # Create and add composite
                if include_composites and files:
                    composite_images = [(name, img) for name, img in files.items()]
                    composite_bytes = create_page_composite(
                        composite_images,
                        page_title=f"{page_name.replace('_', ' ').title()} - Composite View"
                    )
                    zip_file.writestr(f"{page_name}/{page_name}_composite.png", composite_bytes)
        
        zip_buffer.seek(0)
        logger.info("Created export ZIP with %d pages", len(exports))
        return zip_buffer.getvalue()
    
    except Exception as exc:
        logger.exception("Failed to create export ZIP: %s", exc)
        raise


def export_session_state_artifacts(
    session_state: Dict[str, Any],
    page_name: str,
) -> Dict[str, bytes]:
    """Export all visualization artifacts from session state for a given page.

    Args:
        session_state: Streamlit session state dict.
        page_name: Name of the page to export.

    Returns:
        Dict mapping filenames to image bytes.
    """
    exports = {}
    artifacts_key = f"{page_name}_export_artifacts"
    
    if artifacts_key not in session_state:
        logger.warning("No export artifacts found for page: %s", page_name)
        return exports
    
    artifacts = session_state[artifacts_key]
    
    for artifact_name, artifact_data in artifacts.items():
        artifact_type = artifact_data.get('type')
        artifact_obj = artifact_data.get('object')
        
        try:
            if artifact_type == 'plotly':
                img_bytes = export_plotly_chart(artifact_obj, artifact_name)
                exports[f"{artifact_name}.png"] = img_bytes
            
            elif artifact_type == 'dataframe':
                img_bytes = export_dataframe_table(
                    artifact_obj,
                    artifact_name,
                    title=artifact_data.get('title', artifact_name)
                )
                exports[f"{artifact_name}.png"] = img_bytes
            
            elif artifact_type == 'folium':
                img_bytes = export_folium_map(artifact_obj, artifact_name)
                exports[f"{artifact_name}.png"] = img_bytes
            
            elif artifact_type == 'wordcloud':
                img_bytes = export_wordcloud_image(artifact_obj, artifact_name)
                exports[f"{artifact_name}.png"] = img_bytes
            
            else:
                logger.warning("Unknown artifact type: %s for %s", artifact_type, artifact_name)
        
        except Exception as exc:
            logger.error("Failed to export %s: %s", artifact_name, exc)
    
    return exports
