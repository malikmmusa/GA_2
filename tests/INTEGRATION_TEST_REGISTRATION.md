# Integration Test: Image Registration Feature

This document describes the manual integration test procedure for the image registration feature that automatically transfers fovea landmarks from Image 1 to Image 2 using vessel-based alignment.

## Prerequisites

1. Two OCT images from the same patient taken at different times (temporal sequence)
2. Backend server running: `python -m src.api.main` or `./start_api.sh`
3. Frontend server running: `cd src/frontend && npm run dev`
4. Browser open to `http://localhost:3000`

## Test Scenario 1: High Confidence Registration (Success)

### Setup
- Use two OCT images from the same eye, same device, taken weeks/months apart
- Images should have good vessel visibility
- Same anatomical alignment (similar scan angle)

### Steps

1. **Upload Image 1 (Before)**
   - Upload the first temporal image
   - Wait for automatic disc detection (red vertical line)
   - Wait for automatic fovea detection (green circle)
   - Verify the disc and fovea positions are reasonable

2. **Adjust and Confirm Image 1**
   - If needed, drag the disc handles or fovea to adjust positions
   - Click "Confirm Fovea" button for Image 1
   - Wait for GA segmentation to complete

3. **Upload Image 2 (After)**
   - Upload the second temporal image
   - Wait for automatic disc detection
   - Wait for automatic fovea detection
   - **Observe**: Registration should trigger automatically in the background

4. **Verify Registration Success**
   - Check for status message: "✓ Fovea auto-aligned (high confidence: X%)"
   - Confidence should be >= 80%
   - The fovea on Image 2 should now be at the registered position
   - Verify the fovea position on Image 2 aligns anatomically with Image 1

5. **Verify Manual Override Still Works**
   - Before confirming fovea on Image 2, try clicking to adjust the fovea
   - The green marker should move to where you click
   - This confirms the user can still override the auto-registered position

6. **Check Browser Console**
   - Open browser DevTools (F12) → Console tab
   - Look for registration logs:
     ```
     [registration] Starting registration...
     [registration] Result: success, confidence: 0.85
     [registration] High confidence - auto-applying registered fovea
     ```

7. **Verify Backend Logs**
   - Check terminal running the backend API
   - Look for registration service logs:
     ```
     [Registration] En-face shapes: ref=(X, Y), new=(X, Y)
     [Registration] Total matches: N, Good matches after ratio test: M
     [Registration] RANSAC: K inliers out of M matches
     [Registration API] Transformed fovea: (x, y)
     [Registration API] Status: success, Confidence: 0.85
     ```

### Expected Results
- ✅ Registration completes in < 3 seconds
- ✅ Confidence >= 80%
- ✅ Status badge shows "high confidence"
- ✅ Fovea on Image 2 is automatically placed at registered position
- ✅ Fovea position is anatomically sensible (near center, correct side relative to disc)
- ✅ Manual adjustment still works before confirmation
- ✅ Backend logs show successful feature matching (>15 inliers)

---

## Test Scenario 2: Low Confidence Registration

### Setup
- Use two images with:
  - Different scan angles, OR
  - Different zoom levels, OR
  - Lower image quality on one scan

### Steps

1-3. Same as Scenario 1 (upload and confirm Image 1, then upload Image 2)

4. **Verify Low Confidence Behavior**
   - Check for status message: "⚠ Auto-aligned (moderate confidence: X%). Verify position."
   - Confidence should be 40-79%
   - **Observe**: A yellow/orange circle should appear showing the suggested registered position
   - The green circle (current fovea) should remain at the auto-detected position
   - User must choose between the two positions or place manually

5. **Verify Both Markers Visible**
   - Yellow/orange circle = registration suggestion
   - Green circle = independent detection
   - Both should be visible on the canvas

6. **Test Manual Verification**
   - Click near the yellow suggestion to move the green marker there, OR
   - Click elsewhere to place the green marker at a different position
   - Confirm fovea once satisfied with position

### Expected Results
- ✅ Status shows "moderate confidence" with percentage
- ✅ Both markers visible (suggested + current)
- ✅ User can choose which position to use
- ✅ Confidence between 40-79%
- ✅ Backend logs show 8-14 inliers

---

## Test Scenario 3: Registration Failure (Fallback)

### Setup
- Use images that are difficult to register:
  - Different patients, OR
  - Different eyes (OD vs OS), OR
  - Very different scan protocols

### Steps

1-3. Same as Scenario 1

4. **Verify Failure Fallback**
   - No registration status badge should appear, OR
   - Backend console shows: `[registration] Registration failed`
   - Image 2 fovea remains at independently detected position
   - No yellow suggestion marker appears

5. **Verify Independent Detection Works**
   - The fovea on Image 2 should be at the position from independent detection
   - User can adjust manually as normal
   - No registration artifacts or errors

### Expected Results
- ✅ Registration fails gracefully (no error shown to user)
- ✅ Independent fovea detection is used as fallback
- ✅ Application continues to work normally
- ✅ Backend logs show: "Registration failed" or < 8 inliers
- ✅ Confidence < 40%

---

## Test Scenario 4: Registration Not Triggered

### Setup
- Any two images

### Steps

1. **Upload Image 1** without confirming fovea
2. **Upload Image 2**
3. **Verify**: Registration should NOT trigger
   - No "Aligning with Image 1..." message
   - No auto-registration status badge
   - Image 2 uses independent detection only

### Expected Results
- ✅ Registration is not attempted if Image 1 fovea is not confirmed
- ✅ No registration messages or logs
- ✅ Both images processed independently

---

## Test Scenario 5: Reverse Order Upload

### Setup
- Two OCT images from same patient

### Steps

1. **Upload Image 2 (After) first**
   - Process and confirm fovea
2. **Upload Image 1 (Before) second**
   - **Verify**: Registration should NOT trigger (only works "forward" from Before→After)

### Expected Results
- ✅ Registration does not trigger when uploading in reverse order
- ✅ Both images processed independently
- ✅ Application works normally

---

## Automated Test Verification

After manual testing, you can verify the backend unit tests pass:

```bash
cd /Users/musamalik/OCT_Project
pytest tests/test_image_registration.py -v
```

Expected output:
```
tests/test_image_registration.py::TestImageRegistrarService::test_service_initialization PASSED
tests/test_image_registration.py::TestImageRegistrarService::test_enhance_vessels PASSED
tests/test_image_registration.py::TestImageRegistrarService::test_registration_identical_images PASSED
tests/test_image_registration.py::TestImageRegistrarService::test_registration_translated_image PASSED
tests/test_image_registration.py::TestImageRegistrarService::test_registration_different_images PASSED
tests/test_image_registration.py::TestImageRegistrarService::test_transform_landmarks PASSED
tests/test_image_registration.py::TestImageRegistrarService::test_transform_landmarks_with_disc PASSED
tests/test_image_registration.py::TestImageRegistrarService::test_empty_image PASSED
tests/test_image_registration.py::TestImageRegistrarService::test_confidence_scoring PASSED
```

---

## Performance Benchmarks

Expected performance on typical OCT images:

| Metric | Target |
|--------|--------|
| Registration time | < 3 seconds |
| Memory overhead | < 200 MB |
| Feature detection | 500-5000 keypoints per image |
| Good matches (after ratio test) | 50-500 |
| Inliers (after RANSAC) | 15-100 for success |

---

## Troubleshooting

### Issue: Registration always fails
- **Check**: Are images from the same eye?
- **Check**: Do images have visible vessels?
- **Check**: Is en-face region being extracted correctly? (Check backend logs for en-face shapes)
- **Try**: Adjust CLAHE parameters or increase `n_features` in service initialization

### Issue: Low confidence but positions look correct
- **Cause**: Limited overlapping vessel visibility
- **Action**: This is working as intended - user should verify position

### Issue: High confidence but positions look wrong
- **Cause**: False feature matches (rare)
- **Action**: User can manually adjust - this is why manual override is preserved

### Issue: Backend crashes during registration
- **Check**: OpenCV installation is complete
- **Check**: Sufficient memory available
- **Check**: Image dimensions are reasonable (< 10000px)

---

## Sign-off Criteria

✅ All 5 test scenarios pass  
✅ Unit tests pass (pytest)  
✅ Performance within benchmarks  
✅ No console errors or warnings  
✅ Manual override always works  
✅ Failure cases handled gracefully  

**Tested by:** _________________  
**Date:** _________________  
**Notes:** _________________
