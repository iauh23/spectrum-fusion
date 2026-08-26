"""
Positive = the thing we are trying to detect is present
Negative = the absence of that thing
True     = the model is correct
False    = the model is wrong 

Point-in-Box Matching Module is a module that matches ground-truth points with predicted detection boxes.
It is used to compute detection metrics
We specifically use Point-in-Box matching when the ground-truth annotation is a point, but the model prediction is a bounding box.

If a ground-truth point is inside a predicted detection box, it is considered a True Positive (TP).
If a ground-truth point is not inside any predicted detection box, it is considered a False Negative (FN).
If a predicted detection box does not contain any ground-truth points, it is considered a False Positive (FP).  


Ground-truth point: (100, 120) 
point_x = 100 and point_y = 120

Detection box:
    x = 90   left edge of the box
    y = 110  top edge of the box
    w = 30   width of the box
    h = 30   height of the box

Box covers:
    x from 90 to 120 
    y from 110 to 140

Since:
    90 <= 100 <= 120
    110 <= 120 <= 140

The point is inside the box.


Rules:
One ground-truth point can match only one detection.
One detection can match only one ground-truth point.
If multiple boxes contain the same point choose the detection with the highest confidence.

"""

def match_frame_detections(gt_points, detections, modality):
    tp_count = 0  
    fp_count = 0

    match_records = []

    # Filter detections by confidence threshold
    detections =sorted(detections, key = lambda detection :detection['confidence'], reverse=True)
    #key tells what value to use for sorting
    #reverse=True makes the sorting descending instead of ascending without it the sorting happens in ascending order


    available_gt_points = gt_points.copy()  # Create a copy of gt_points to keep track of unmatched points

    for detection in detections:
        x,y,w,h = detection['box']
        cx = x + w / 2.0
        cy = y + h / 2.0

        gt_inside =[]

        for px, py in available_gt_points:
            if x <= px <= (x + w) and y <= py <= (y + h):
                gt_inside.append((px, py))


        #Rule: FP
        if not gt_inside:
            fp_count += 1
            continue

        #Rule: TP
        if gt_inside:

            best_gt_point = min(gt_inside,key=lambda p: ((p[0] - cx) ** 2 + (p[1] - cy) ** 2) ** 0.5, )

            distance = ((best_gt_point[0] - cx) ** 2 + (best_gt_point[1] - cy) ** 2) ** 0.5
            tp_count += 1
            available_gt_points.remove(best_gt_point)  # Remove the matched point from available_gt_points

            match_records.append({
                'gt_point': best_gt_point,
                'detection_box': detection['box'],
                'confidence': detection['confidence'],
                'modality': modality,
                'distance': distance
            })
    # Rule: FN   
    fn_count = len(available_gt_points)

    return {
        "tp": tp_count,
        "fp": fp_count,
        "fn": fn_count,
        "match_records": match_records,
        "unmatched_gt_points": available_gt_points,
    }