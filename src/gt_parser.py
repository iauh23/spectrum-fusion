"""
XML GT Parser - parses person-point annotations from XML files.

XML schema (per file):
    <annotation>
        <size><width>640</width><height>512</height>...</size>
        <object>
            <name>person</name>
            <point><x>100</x><y>200</y></point>
        </object>
        ...

Output: {image_id: [(x, y), ...]}
    - image_id = XML filename without extension (e.g., "1R.xml" -> "1R")
    - (x, y) = integer pixel coordinates of a person

Pairing rule:
    RGB:      data/rgb/{N}.jpg       e.g. data/rgb/1.jpg
    Thermal:  data/infrared/{N}R.jpg e.g. data/infrared/1R.jpg
    GT:       data/GT/{N}R.xml       e.g. data/GT/1R.xml
    The number N is shared. "R" suffix = thermal + GT.
"""

from pathlib import Path
import xml.etree.ElementTree as ET
import json
import os

def gt_box(xml_input=r'C:\Users\ia443\Desktop\Project\data\GT'):
     xml_folder = Path(xml_input)

     # Cache: return cached result if it exists and is newer than all XML files
     cache_path = xml_folder / "gt_cache.json"
     if cache_path.exists():
          cache_mtime = os.path.getmtime(cache_path)
          xml_files = list(xml_folder.glob("*.xml"))
          if xml_files:
               newest_xml = max(os.path.getmtime(f) for f in xml_files)
               if cache_mtime > newest_xml:
                    with open(cache_path, "r") as f:
                         cached = json.load(f)
                    # JSON stores lists, not tuples - convert back
                    return {k: [tuple(p) for p in v] for k, v in cached.items()}

     gt_dict = {}

     for xml_file in xml_folder.glob('*.xml'):
          # Malformed XML: catch parse errors and skip the file
          try:
               tree = ET.parse(xml_file)
               root = tree.getroot()
          except ET.ParseError as e:
               print(f"Warning: {xml_file.name} is malformed XML: {e}")
               continue

          image_id = xml_file.stem
          co_ordinates = []

          for obj in root.findall("object"):
              name = obj.findtext("name")

              if name == 'person':
                   point = obj.find('point')

                   if point is None:
                        print(f"Warning: {xml_file.name} has <object> with name 'person' but no <point> tag")
                        continue

                   x = point.findtext("x")
                   y = point.findtext("y")

                   if x is None or y is None:
                        print(f"Warning: {xml_file.name} has incomplete point data (missing x or y)")
                        continue

                   x = int(x)
                   y = int(y)

                   # Out-of-bounds check (all images are 640x512)
                   if x < 0 or x > 640 or y < 0 or y > 512:
                        print(f"Warning: {xml_file.name} has out-of-bounds point ({x}, {y})")

                   co_ordinates.append((x, y))

          # Zero-person frames: log warning
          if len(co_ordinates) == 0:
               print(f"Warning: {xml_file.name} has zero persons")

          # Duplicate points: detect and warn
          if len(co_ordinates) != len(set(co_ordinates)):
               print(f"Warning: {xml_file.name} has duplicate points")

          gt_dict[image_id] = co_ordinates

     # Cache: save parsed result for next run
     with open(cache_path, "w") as f:
          json.dump(gt_dict, f)

     return gt_dict
