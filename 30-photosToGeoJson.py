import os
import json
import easyocr
import geojson
import toml
import ollama
from exif import Image as ExifImage
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

enable_ocr = False
enable_llm = True
llm_model = "llama3.2-vision"
llm_model = "minicpm-v"
llm_model = "benzie/llava-phi-3"
llm_model = "moondream"

llm_prompt_top = "Extract and correct the french text from this image. Do not add any additional text. Just output the text you manage to read."
llm_prompt_dest = "Extract and correct the french text from this image. Do not add any additional text. Just output the text you manage to read."

def convert_uuid_to_url(uuid_string):
    base_url = "https://panoramax.openstreetmap.fr/images"
    # Split the UUID string into segments
    segments = [uuid_string[i:i+2] for i in range(0, 8, 2)] + [uuid_string[9:]]
    # Construct the URL
    url = f"{base_url}/{segments[0]}/{segments[1]}/{segments[2]}/{segments[3]}/{segments[4]}.jpg"
    return url

def getPanoramax(filename):
    # Load and parse the TOML file
    toml_file_path = "./photos/HikingSigns/_geovisio.toml"
    toml_data = toml.load(toml_file_path)

    # Search for the filename in the TOML data
    for section in toml_data.values():
        if isinstance(section, dict):
            pictures = section.get('pictures', {})
            for pic_key, pic_data in pictures.items():
                if pic_data.get('path') == filename:
                    pic_id = pic_data.get('id')
                    hd_href = convert_uuid_to_url(pic_id)
                    return pic_id, hd_href

    print(f"Error: File {filename} not found in the TOML data.")
    return None

def extract_exif_data(image_path):
    with open(image_path, 'rb') as image_file:
        image = ExifImage(image_file)
        if image.has_exif:
            try:
                lat = image.gps_latitude
                lon = image.gps_longitude
                lat = lat[0] + lat[1] / 60 + lat[2] / 3600
                lon = lon[0] + lon[1] / 60 + lon[2] / 3600
                return lat, lon
            except AttributeError as e:
                print(f"Error: File {image_path} miss lat/lon in exif data. {e}")
                return None, None
        else:
            print(f"Error: File {image_path} do not have exif data.")
            return None, None

def get_ocr_text(reader, image_path):
    return " \n".join(reader.readtext(image_path, detail=0))

def get_llm_text(image_path, prompt):
    res = ollama.chat(
        model=llm_model,
        messages= [{
            "role": "user",
            "content": prompt,
            "images": [image_path]
        }],
    )

    return res["message"]["content"]

def create_geojson_feature(lat, lon, filename, properties):
    feature = {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [lon, lat]
        },
        "properties": {
            "filename": filename,
            **properties
        }
    }
    return feature

def process_image(reader, filename, photo_folder, crop_top_folder, crop_destination_folder):
    image_path = os.path.join(photo_folder, filename)
    lat, lon = extract_exif_data(image_path)
    if lat and lon:
        properties = {
            "tourism": "information",
            "information": "guidepost",
            "hiking": "yes"
        }
        base_filename = os.path.splitext(filename)[0]

        ocr_names, llm_names = [], []
        index, index_ocr, index_llm = 1, 1, 1
        while True:
            top_image_path = os.path.join(crop_top_folder, f"{base_filename}_top_{index}.jpg")
            if not os.path.exists(top_image_path):
                break
            if enable_ocr:
                easy_text = get_ocr_text(reader, top_image_path)
                if easy_text.strip():
                    properties["name:ocr:{index_ocr}"] = easy_text
                    ocr_names.append(easy_text)
                    index_ocr += 1
            if enable_llm:
                llm_text = get_llm_text(top_image_path, llm_prompt_top)
                if llm_text.strip():
                    properties[f"name:llm:{index_llm}"] = llm_text
                    llm_names.append(llm_text)
                    index_llm += 1
            index += 1
            

        if len(ocr_names) > 0:
            properties["name:ocr:all"] = " ;\n".join(ocr_names)

        if len(llm_names) > 0:
            properties["name:llm:all"] = " ;\n".join(llm_names)

        ocr_dests, llm_dests = [], []
        index, index_ocr, index_llm = 1, 1, 1
        while True:
            dest_image_path = os.path.join(crop_destination_folder, f"{base_filename}_destination_{index}.jpg")
            if not os.path.exists(dest_image_path):
                break
            if enable_ocr:
                easy_text = get_ocr_text(reader, dest_image_path)
                if easy_text.strip():
                    properties[f"dest:ocr:{index_ocr}"] = easy_text
                    ocr_dests.append(easy_text)
                    index_ocr += 1
            if enable_llm:
                llm_text = get_llm_text(dest_image_path, llm_prompt_dest)
                if llm_text.strip():
                    properties[f"dest:llm:{index_llm}"] = llm_text
                    llm_dests.append(llm_text)
                    index_llm += 1
            index += 1

        if len(ocr_dests) > 0:
            properties["dest:ocr:all"] = " ;\n".join(ocr_dests)

        if len(llm_dests) > 0:
            properties["dest:llm:all"] = " ;\n".join(llm_dests)

        id, hd_href = getPanoramax(filename)
        if id:
            properties["panoramax"] = id
            properties["panoramax:hd_href"] = hd_href

        return create_geojson_feature(lat, lon, filename, properties)

def main():
    photo_folder = "./photos/HikingSigns"
    crop_top_folder = "./crop/top"
    crop_destination_folder = "./crop/destination"
    output_file = "hikingSigns.geojson"
    geojson_features = []

    if os.path.exists(output_file):
        os.remove(output_file)

    filenames = [f for f in os.listdir(photo_folder) if f.endswith(".jpg")]

    reader = None
    if enable_ocr:
        reader = easyocr.Reader(['fr'], gpu=True)  # Initialize EasyOCR reader once
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {executor.submit(process_image, reader, filename, photo_folder, crop_top_folder, crop_destination_folder): filename for filename in filenames}
            for future in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                feature = future.result()
                if feature:
                    geojson_features.append(feature)
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(geojson.FeatureCollection(geojson_features), f, indent=2, ensure_ascii=False)

    elif enable_llm:
        for filename in tqdm(filenames, total=len(filenames), desc="Processing images"):
            feature = process_image(reader, filename, photo_folder, crop_top_folder, crop_destination_folder)
            if feature:
                geojson_features.append(feature)
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(geojson.FeatureCollection(geojson_features), f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
