"""
Precision tell how many predictions made by the model are actually correct
Recall tells how many of the actual positives were captured by the model
F1 takes harmonic mean of precision and recall to tell how model balnaces accuracy and performance  

"""
import math 

def compute_frame_metric(tp, fp, fn, match_records):
  """
  works with 100 images one by one to give each photo its score because at the end 
result will be saved as a csv file
  """
  distances = []
  for record in match_records:
     distances.append(record['distance'])
  
  if tp + fp == 0:
    precision = None
  else:
    precision = tp / (tp + fp)  

  if tp + fn == 0:
    recall = None

  else:
    recall = tp / (tp + fn)  

  if precision is not None and recall is not None:
    if precision + recall == 0:
      f1 = 0.0
    else:
       f1 = 2 * (precision * recall) / (precision + recall) 

  else:
    f1 = None

  return {
    "precision": precision,
    "recall": recall,
    "f1": f1,
    "tp": tp,  
    "fp": fp,
    "fn": fn,
    "distances": distances
  }

"""
As my gt are points and my predictions are boxes.
Also the number of people per image varies
So there are two approcaches:
a. Frame weighted averaging where each frame contributes equally. Every scene is equally important to get right
b. Person Weighted averaging where frame with more people contribute more. Every person is important to count
My approach is to use frame weighted averaging as it is more important to get the scene right than to get the number of people right.
"""

def compute_metrics(frame_results):

  all_distances = []
  precisions = []
  recalls = []
  f1s = []
  total_tp = 0
  total_fp = 0
  total_fn = 0

  for result in frame_results:
     all_distances.extend(result['distances'])
     total_tp += result['tp']
     total_fp += result['fp']
     total_fn += result['fn']

     # Frame-weighted: collect each frame's score, skipping N/A (None)
     if result['precision'] is not None:
        precisions.append(result['precision'])
     if result['recall'] is not None:
        recalls.append(result['recall'])
     if result['f1'] is not None:
        f1s.append(result['f1'])

  # MAE: mean of pooled distances
  if len(all_distances) == 0:
      mae = None
  else:
      mae = sum(all_distances) / len(all_distances)

  # RMSE: root mean squared of pooled distances
  squared_distances = [d ** 2 for d in all_distances] 
  rmse = math.sqrt(sum(squared_distances) / len(squared_distances)) if squared_distances else None    

  # Macro (frame-weighted) averages, with empty-list guards
  precision = sum(precisions) / len(precisions) if precisions else None
  recall = sum(recalls) / len(recalls) if recalls else None
  f1 = sum(f1s) / len(f1s) if f1s else None

  return {
      "precision": precision,
      "recall": recall,
      "f1": f1,
      "mae": mae,
      "rmse": rmse,
      "total_tp": total_tp,
      "total_fp": total_fp,
      "total_fn": total_fn
    }
