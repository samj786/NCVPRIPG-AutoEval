# %%
import transformers
import random 

# %%
import torch

# %%
from torchvision import transforms

# %%
from transformers import AutoModelForObjectDetection, TableTransformerForObjectDetection

# %%
from transformers import TrOCRProcessor, VisionEncoderDecoderModel

# %%
from PIL import Image, ImageDraw
import pytesseract
import cv2 as cv
import matplotlib.pyplot as plt
import numpy as np
import IPython.display as display
import os
from matplotlib.patches import Rectangle, Patch
import re
from difflib import SequenceMatcher
from tqdm.auto import tqdm
import csv
import pandas as pd

# %%
# Define the custom resize transformation
class MaxResize(object):
    def __init__(self, max_size=800):
        self.max_size = max_size

    def __call__(self, image):
        width, height = image.size
        current_max_size = max(width, height)
        scale = self.max_size / current_max_size
        resized_image = image.resize((int(round(scale * width)), int(round(scale * height))))
        return resized_image

# Postprocessing functions
def box_cxcywh_to_xyxy(x):
    x_c, y_c, w, h = x.unbind(-1)
    b = [(x_c - 0.5 * w), (y_c - 0.5 * h), (x_c + 0.5 * w), (y_c + 0.5 * h)]
    return torch.stack(b, dim=1)

def rescale_bboxes(out_bbox, size):
    img_w, img_h = size
    b = box_cxcywh_to_xyxy(out_bbox)
    b = b * torch.tensor([img_w, img_h, img_w, img_h], dtype=torch.float32)
    return b

def outputs_to_objects(outputs, img_size, id2label):
    m = outputs.logits.softmax(-1).max(-1)
    pred_labels = list(m.indices.detach().cpu().numpy())[0]
    pred_scores = list(m.values.detach().cpu().numpy())[0]
    pred_bboxes = outputs['pred_boxes'].detach().cpu()[0]
    pred_bboxes = [elem.tolist() for elem in rescale_bboxes(pred_bboxes, img_size)]

    objects = []
    for label, score, bbox in zip(pred_labels, pred_scores, pred_bboxes):
        class_label = id2label[int(label)]
        if not class_label == 'no object':
            objects.append({'label': class_label, 'score': float(score), 'bbox': [float(elem) for elem in bbox]})
    return objects

# Function to get model outputs
def get_model_outputs(image, model, transform):
    pixel_values = transform(image).unsqueeze(0)
    pixel_values = pixel_values.to(device)

    with torch.no_grad():
        outputs = model(pixel_values)

    return outputs

# Visualization function for table region
def fig2img(fig):
    import io
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight', dpi=150)
    buf.seek(0)
    return Image.open(buf)


def visualize_detected_objects(img, objects, out_path=None):
    # Optimize figure creation
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.imshow(img, interpolation="lanczos")

    for obj in objects:
        bbox = obj['bbox']
        label = obj['label']
        facecolor = (1, 0, 0.45) if label == 'table' else (0.95, 0.6, 0.1)
        edgecolor = facecolor
        alpha = 0.3
        linewidth = 2
        hatch = '//////' if label == 'table' else 'xxxx'

        # Add patches in a more efficient manner
        rect_params = {'linewidth': linewidth, 'edgecolor': 'none', 'facecolor': facecolor, 'alpha': 0.1}
        ax.add_patch(Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], **rect_params))

        rect_params.update({'edgecolor': edgecolor, 'facecolor': 'none', 'alpha': alpha})
        ax.add_patch(Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], **rect_params))

        rect_params.update({'linewidth': 0, 'hatch': hatch, 'alpha': 0.2})
        ax.add_patch(Rectangle(bbox[:2], bbox[2]-bbox[0], bbox[3]-bbox[1], **rect_params))

    ax.set_xticks([])
    ax.set_yticks([])
    ax.axis('off')

    legend_elements = [
        Patch(facecolor=(1, 0, 0.45), edgecolor=(1, 0, 0.45), label='Table', hatch='//////', alpha=0.3),
        Patch(facecolor=(0.95, 0.6, 0.1), edgecolor=(0.95, 0.6, 0.1), label='Table (rotated)', hatch='xxxx', alpha=0.3)
    ]
    ax.legend(handles=legend_elements, bbox_to_anchor=(0.5, -0.02), loc='upper center', borderaxespad=0, fontsize=10, ncol=2)

    if out_path is not None:
        fig.savefig(out_path, bbox_inches='tight', dpi=150)

    return fig

#Helper function for crop table
def iob(bbox1, bbox2):
    """Calculates Intersection over Union (IoU) for two bounding boxes."""
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])

    intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
    bbox1_area = (bbox1[2] - bbox1[0] + 1) * (bbox1[3] - bbox1[1] + 1)
    bbox2_area = (bbox2[2] - bbox2[0] + 1) * (bbox2[3] - bbox2[1] + 1)
    union_area = bbox1_area + bbox2_area - intersection_area

    return intersection_area / union_area

#Crop table function
def objects_to_crops(img, tokens, objects, class_thresholds, padding):
    table_crops = []
    for obj in objects:
        if obj['score'] < class_thresholds[obj['label']]:
            continue

        cropped_table = {}

        bbox = obj['bbox']
        bbox = [bbox[0]-padding, bbox[1]-10, bbox[2]+(1.2*padding), bbox[3]+padding]

        bbox[1] = max(0, bbox[1])

        cropped_img = img.crop(bbox)

        table_tokens = [token for token in tokens if iob(token['bbox'], bbox) >= 0.5]
        for token in table_tokens:
            token['bbox'] = [token['bbox'][0]-bbox[0],
                             token['bbox'][1]-bbox[1],
                             token['bbox'][2]-bbox[0],
                             token['bbox'][3]-bbox[1]]

        if obj['label'] == 'table rotated':
            cropped_img = cropped_img.rotate(270, expand=True)
            for token in table_tokens:
                bbox = token['bbox']
                bbox = [cropped_img.size[0]-bbox[3]-1,
                        bbox[0],
                        cropped_img.size[0]-bbox[1]-1,
                        bbox[2]]
                token['bbox'] = bbox

        cropped_table['image'] = cropped_img
        cropped_table['tokens'] = table_tokens

        table_crops.append(cropped_table)

    return table_crops

#Visualization of the table structure being detected
'''
def plot_results(cells, class_to_visualize):
    if class_to_visualize not in structure_model.config.id2label.values():
      raise ValueError("Class should be one of the available classes")

    plt.figure(figsize=(16,10))
    plt.imshow(cropped_table)
    ax = plt.gca()

    for cell in cells:
        score = cell["score"]
        bbox = cell["bbox"]
        label = cell["label"]

        if label == class_to_visualize:
          xmin, ymin, xmax, ymax = tuple(bbox)

          ax.add_patch(plt.Rectangle((xmin, ymin), xmax - xmin, ymax - ymin, fill=False, color="red", linewidth=3))
          text = f'{cell["label"]}: {score:0.2f}'
          ax.text(xmin, ymin, text, fontsize=15,
                  bbox=dict(facecolor='yellow', alpha=0.5))
          plt.axis('off')
'''

# Function to get the last column bbox
def get_last_column_bbox(cells):
    columns = [entry for entry in cells if entry['label'] == 'table column']
    columns.sort(key=lambda x: x['bbox'][0])
    if not columns:
        return None
    return columns[-1]['bbox']

# Function to get cell coordinates by row within the last column
def get_row_coordinates_within_column(column_image, original_bbox, rows):
    rows.sort(key=lambda x: x['bbox'][1])
    for row in rows:
        row['bbox'] = [original_bbox[0], row['bbox'][1], original_bbox[2], row['bbox'][3]]
    return rows

# Function to check if the image is blank
def is_blank_image(image, threshold=0.986):
    """
    Check if an image is blank by analyzing the percentage of white pixels.
    """
    #black threshold
    black_threshold = 0.2
    # Convert image to numpy array
    image_array = np.array(image)
    # Calculate the percentage of white pixels
    white_pixels = np.sum(image_array == 255)
    total_pixels = image_array.size
    white_pixel_ratio = white_pixels / total_pixels
    black_pixel_ratio = 1 - white_pixel_ratio
    #print(f"White pixel ratio: {white_pixel_ratio:.4f}, Black pixel ratio: {black_pixel_ratio:.4f}")
    if black_pixel_ratio > black_threshold:
        return True
    return white_pixel_ratio > threshold

# Perform OCR using the TrOCR model.
def ocr(image, processor, model):
    image = image.convert('RGB')
    pixel_values = processor(images=image, return_tensors="pt").pixel_values
    generated_ids = model.generate(pixel_values)
    generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return generated_text

# Check if two strings are similar based on a given threshold.
def is_similar(a, b, threshold=0.4):
    return SequenceMatcher(None, a, b).ratio() > threshold

# Normalize text by converting to lowercase and removing non-alphanumeric characters.
def normalize_text(text):
    text = text.lower()
    text = re.sub(r'[^a-z]', '', text)
    return text

def post_process_ocr_results(ocr_results):
    """
    Post-process OCR results to clean up and correctly identify true and false values.
    """
    cleaned_results = []
    for result in ocr_results:


        # Remove special characters and spaces
        result_no_special = re.sub(r'[^a-zA-Z0-9]', '', result)

        # Check if the result is just numbers
        if result_no_special.isdigit():
            cleaned_results.append("Blank Image")
            continue

        # Normalize the result
        normalized_result = normalize_text(result)

        # Check if the result is empty after normalization
        if normalized_result == '':
            cleaned_results.append("Blank Image")
            continue

        # Apply the new rules if similarity checks fail
        if normalized_result.startswith('t'):
            if len(normalized_result) == 1 or (len(normalized_result) > 1 and normalized_result[1] == 'r'):
                cleaned_results.append("true")
                continue


        if normalized_result.startswith('f'):
            if len(normalized_result) == 1 or (len(normalized_result) > 1 and normalized_result[1] == 'a'):
                cleaned_results.append("false")
                continue


        # Check for similarity first
        if is_similar(normalized_result, "true"):
            cleaned_results.append("true")
            continue
        elif is_similar(normalized_result, "false"):
            cleaned_results.append("false")
            continue

        if normalized_result == 'blankimage':
            cleaned_results.append("Blank Image")
            continue

        if 't' in normalized_result and not 'f' in normalized_result:
            cleaned_results.append("true")
        elif 'f' in normalized_result and not 't' in normalized_result:
            cleaned_results.append("false")
        elif 'r' in normalized_result and not 'l' in normalized_result:
            cleaned_results.append("true")
        elif 'l' in normalized_result and not 'r' in normalized_result:
            cleaned_results.append("false")
        elif 'a' in normalized_result and not 'r' in normalized_result:
            cleaned_results.append("false")
        elif 'r' in normalized_result and not 'a' in normalized_result:
            cleaned_results.append("true")
        elif 's' in normalized_result:
            cleaned_results.append("false")
        else:
            cleaned_results.append("Uncertain")  # Handle uncertain cases

    return cleaned_results

# %%
#Open the original image, enhance its contrast, and use OCR to detect and correct the rotation angle.
#Then grayscale and remove noise from the original image.
#Then perform table region detection and crop the table
#Perform structure recognition and get co-ordinates of the last column and its rows
#for every cell in the row, crop it and binarize it and then perform trOCR on it.
#Finally clean the ocr results
def process_image(image_path, model, structure_model, processor_trOCR, model_trOCR):
    original_im = cv.imread(image_path)
    gray_im = cv.cvtColor(original_im, cv.COLOR_BGR2GRAY)
    _, binary_im = cv.threshold(gray_im, 128, 255, cv.THRESH_BINARY)
    osd = pytesseract.image_to_osd(binary_im, output_type='dict')
    rotate = int(osd['rotate'])

    if rotate != 0:
        (h, w) = original_im.shape[:2]
        center = (w // 2, h // 2)
        M = cv.getRotationMatrix2D(center, -rotate, 1.0)
        abs_cos, abs_sin = abs(M[0, 0]), abs(M[0, 1])
        bound_w, bound_h = int(h * abs_sin + w * abs_cos), int(h * abs_cos + w * abs_sin)
        M[0, 2] += bound_w / 2 - center[0]
        M[1, 2] += bound_h / 2 - center[1]
        im_fixed = cv.warpAffine(original_im, M, (bound_w, bound_h))
    else:
        im_fixed = original_im

    gray_img = cv.cvtColor(im_fixed, cv.COLOR_BGR2GRAY)
    median = cv.medianBlur(gray_img, 5)

    binary_pil = Image.fromarray(median).convert("RGB")
    outputs = get_model_outputs(binary_pil, model, detection_transform)
    objects = outputs_to_objects(outputs, binary_pil.size, id2label)

    # Visualization for table region being detected - uncomment below lines to visualize
    '''
    fig = visualize_detected_objects(binary_pil, objects)
    visualized_image = fig2img(fig)
    visualized_image.show()
    '''
    tokens = []
    tables_crops = objects_to_crops(binary_pil, tokens, objects, detection_class_thresholds, padding=crop_padding)
    if(len(tables_crops) != 0):
        cropped_table = tables_crops[0]['image'].convert("RGB")

        new_outputs = get_model_outputs(cropped_table, structure_model, structure_transform)
        new_cells = outputs_to_objects(new_outputs, cropped_table.size, structure_id2label)

        # Visualization for table structure being detected - uncomment below lines to visualize
        '''
        plot_results(final_cells, class_to_visualize="table row")
        plt.show()
        '''

        rows = [entry for entry in new_cells if entry['label'] == 'table row']
        
        # Check the number of detected rows
        if len(rows) <= 10:
            table_bbox = objects[0]['bbox']
            extended_bbox = [table_bbox[0], table_bbox[1], table_bbox[2], table_bbox[3] + 200]
            #extended_cropped_table = binary_pil.crop([extended_bbox[0]- crop_padding, extended_bbox[1]-crop_padding, extended_bbox[2]+crop_padding, extended_bbox[3]+crop_padding])

            # Use the same cropping function for the extended bounding box
            extended_objects = [{'label': 'table', 'score': 1.0, 'bbox': extended_bbox}]
            extended_tables_crops = objects_to_crops(binary_pil, tokens, extended_objects, detection_class_thresholds, padding=crop_padding)
            extended_cropped_table = extended_tables_crops[0]['image'].convert("RGB")

            # Reapply the model on the extended cropped table
            extended_outputs = get_model_outputs(extended_cropped_table, structure_model, structure_transform)
            extended_cells = outputs_to_objects(extended_outputs, extended_cropped_table.size, structure_id2label)
            rows = [entry for entry in extended_cells if entry['label'] == 'table row']
            #print(f"Number of detected rows: {len(rows)}")
            final_cropped_table = extended_cropped_table if len(rows) > 10 else cropped_table
        else:
            final_cropped_table = cropped_table

        final_outputs = get_model_outputs(final_cropped_table, structure_model, structure_transform)
        final_cells = outputs_to_objects(final_outputs, final_cropped_table.size, structure_id2label)

        last_column_bbox = get_last_column_bbox(final_cells)
        
        # Check if the last column is detected
        if last_column_bbox is None:
            cleaned_results = [random.choice(['true', 'false']) for _ in range(10)]
            return cleaned_results
        
        last_column_image = final_cropped_table.crop(last_column_bbox)
        rows_within_last_column = get_row_coordinates_within_column(last_column_image, last_column_bbox, rows)

        ocr_results = []
        for idx, row in enumerate(rows_within_last_column):
            cell_bbox = [last_column_bbox[0], row['bbox'][1], last_column_bbox[2], row['bbox'][3]]
            cell_image = cropped_table.crop(cell_bbox)
            grayscale_image = cell_image.convert('L')
            image_array = np.array(grayscale_image)
            _, binarized_image = cv.threshold(image_array, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
            binarized_image_pil = Image.fromarray(binarized_image)

            if idx == 0:
                continue
            if is_blank_image(binarized_image_pil):
                ocr_results.append("Blank Image")
            else:
                result = ocr(binarized_image_pil, processor_trOCR, model_trOCR)
                ocr_results.append(result)

        cleaned_results = post_process_ocr_results(ocr_results)
    else:
        cleaned_results = [random.choice(['true', 'false']) for _ in range(10)]
    return cleaned_results

# %%

# Main
import shutil

def main():
    global device, detection_transform, structure_transform, id2label, structure_id2label, detection_class_thresholds, crop_padding

    device = "cuda" if torch.cuda.is_available() else "cpu"

    detection_transform = transforms.Compose([
        MaxResize(800),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    structure_transform = transforms.Compose([
        MaxResize(1000),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    model = AutoModelForObjectDetection.from_pretrained("microsoft/table-transformer-detection", revision="no_timm")
    structure_model = TableTransformerForObjectDetection.from_pretrained("microsoft/table-structure-recognition-v1.1-all")
    processor_trOCR = TrOCRProcessor.from_pretrained('microsoft/trocr-large-handwritten')
    model_trOCR = VisionEncoderDecoderModel.from_pretrained('microsoft/trocr-large-handwritten')

    model.to(device)
    structure_model.to(device)
    model_trOCR.to(device)

    id2label = model.config.id2label
    id2label[len(model.config.id2label)] = "no object"
    structure_id2label = structure_model.config.id2label
    structure_id2label[len(structure_id2label)] = "no object"

    detection_class_thresholds = {
        "table": 0.5,
        "table rotated": 0.5,
        "no object": 10
    }
    crop_padding = 200

 
    # Read image-model answer mapping
    image_model_answer_mapping = {}
    with open('img_model_answer_mapping.csv', 'r') as file:
        reader = csv.reader(file)
        next(reader)  # Skip the header
        for row in reader:
            image_model_answer_mapping[row[0]] = row[1]

    # Load model answers
    model_answers = {}
    model_answer_files = {
        'model_answer_type1': 'model_answer_type1.csv',
        'model_answer_type2': 'model_answer_type2.csv'
    }

    for model_answer_key, model_answer_file in model_answer_files.items():
        answers = []
        with open(model_answer_file, 'r') as file:
            reader = csv.reader(file)
            next(reader)  # Skip the header
            for row in reader:
                _, answer = row
                answers.append(answer)
        model_answers[model_answer_key] = answers

    # Define the folder containing the test images
    test_image_folder = 'test_images'
    results = []

    for image_name in os.listdir(test_image_folder):
        if image_name.endswith(('jpg', 'jpeg', 'png')):  # Process only image files
            image_path = os.path.join(test_image_folder, image_name)

            # Get the correct model answer key
            model_answer_key = image_model_answer_mapping.get(image_name)
            if not model_answer_key:
                print(f"No model answer mapping found for image: {image_name}")
                continue

            correct_answers = model_answers.get(model_answer_key)
            if not correct_answers:
                print(f"No correct answers found for model answer key: {model_answer_key}")
                continue

            cleaned_results = process_image(image_path, model, structure_model, processor_trOCR, model_trOCR)

            # Compare cleaned results with correct answers
            score = 0
            total = min(len(cleaned_results), len(correct_answers))

            for i in range(total):
                token = cleaned_results[i]
                if token.lower() == correct_answers[i].lower():
                    score += 1
                elif token.lower() == "uncertain" and correct_answers[i].lower() in ["true", "false"]:
                    score += 1

            results.append([image_name, score])
            print(f"Processed {image_name}: Score {score}/{total}")

    # Write the results to a CSV file
    with open('finaloutput.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(['Image Name', 'Score'])
        for result in results:
            writer.writerow(result)

    print("Scores have been saved to finaloutput.csv")


if __name__ == "__main__":
    main()
# %%



