/*
* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
*
* Licensed under the Apache License, Version 2.0 (the "License");
* you may not use this file except in compliance with the License.
* You may obtain a copy of the License at
*
*       http://www.apache.org/licenses/LICENSE-2.0
*
* Unless required by applicable law or agreed to in writing, software
* distributed under the License is distributed on an "AS IS" BASIS,
* WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
* See the License for the specific language governing permissions and
* limitations under the License.
*/

package pp.facerecognizer;

import android.content.ClipData;
import android.content.Intent;
import android.content.SharedPreferences;
import android.content.res.AssetManager;
import android.graphics.Bitmap;
import android.graphics.Bitmap.Config;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Matrix;
import android.graphics.Paint;
import android.graphics.Typeface;
import android.media.ImageReader.OnImageAvailableListener;
import android.net.Uri;
import android.os.Bundle;
import android.os.SystemClock;
import android.preference.PreferenceManager;
import android.util.Log;
import android.util.Size;
import android.util.TypedValue;
import android.view.View;
import android.widget.EditText;
import android.widget.FrameLayout;

import com.google.android.material.floatingactionbutton.FloatingActionButton;
import com.google.android.material.snackbar.Snackbar;

import org.sofwerx.ogc.sos.SensorMeasurement;
import org.sofwerx.ogc.sos.SensorMeasurementTime;
import org.sofwerx.ogc.sos.SensorResultTemplateField;
import org.sofwerx.ogc.sos.SensorTextResultTemplateField;
import org.sofwerx.ogc.sos.SosSensor;
import org.sofwerx.ogc.sos.SosService;

import java.io.File;
import java.io.StringWriter;
import java.util.ArrayList;
import java.util.Collections;
import java.util.List;
import java.util.Random;
import java.util.Vector;

import androidx.appcompat.app.AlertDialog;
import pp.facerecognizer.env.BorderedText;
import pp.facerecognizer.env.FileUtils;
import pp.facerecognizer.env.ImageUtils;
import pp.facerecognizer.env.Logger;
import pp.facerecognizer.tracking.MultiBoxTracker;
import pp.objectdetection.customview.OverlayView;
import pp.objectdetection.customview.OverlayView.DrawCallback;
import pp.objectdetection.tflite.DetectionClassifier;
import pp.objectdetection.tflite.TFLiteObjectDetectionAPIModel;




/**
* An activity that uses a TensorFlowMultiBoxDetector and ObjectTracker to detect and then track
* objects.
*/
public class MainActivity extends CameraActivity implements OnImageAvailableListener {

    // Which detection model to use: by default uses Tensorflow Object Detection API frozen
    // checkpoints.
    private enum DetectorMode {
        TF_OD_API;
    }

    private final static String TAG = "AFD";
    private static final Logger LOGGER = new Logger();

    // Configuration values for the prepackaged SSD model.
    private static final int TF_OD_API_INPUT_SIZE = 300;
    private static final boolean TF_OD_API_IS_QUANTIZED = true;
    private static final String TF_OD_API_MODEL_FILE = "detect.tflite";
    private static final String TF_OD_API_LABELS_FILE = "file:///android_asset/labelmap.txt";
    private static final DetectorMode MODE = DetectorMode.TF_OD_API;
    // Minimum detection confidence to track a detection.
    private static final float MINIMUM_CONFIDENCE_TF_OD_API = 0.5f;

    private static final int FACE_SIZE = 160;
    private static final int CROP_SIZE = 300;

    private static final Size DESIRED_PREVIEW_SIZE = new Size(640, 480);

    private static final boolean SAVE_PREVIEW_BITMAP = false;
    private static final float TEXT_SIZE_DIP = 10;

    private Integer sensorOrientation;

    private Classifier classifier;
    private DetectionClassifier detectionClassifier;

    private long lastProcessingTimeMs;
    private Bitmap rgbFrameBitmap = null;
    private Bitmap croppedBitmap = null;
    private Bitmap cropCopyBitmap = null;

    private boolean computingDetection = false;

    private long timestamp = 0;

    private Matrix frameToCropTransform;
    private Matrix cropToFrameTransform;

    private MultiBoxTracker tracker;

    private byte[] luminanceCopy;

    private BorderedText borderedText;

    private Snackbar initSnackbar;
    private Snackbar trainSnackbar;
    private FloatingActionButton button;
    private FloatingActionButton buttonSettings;

    private boolean initialized = false;
    private boolean training = false;

    //SOS
    private boolean reportOverSos;
    private SosService sosService;
    private SosSensor sosSensor;
    private SensorMeasurementTime sensorMeasurementTime;
    private SensorMeasurement sensorMeasurementName;
    private SensorMeasurement sensorMeasurementConfidence;

    private String sosURL, callsign, sensorId, sosUsername, sosPassword;
    private float sosConfidenceToReport;
    private long maxReportingInterval;
    private long nextAvailableReportingTime = Long.MIN_VALUE;

    @Override
    protected void onCreate(final Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);

        FrameLayout container = findViewById(R.id.container);
        initSnackbar = Snackbar.make(container, "Initializing...", Snackbar.LENGTH_INDEFINITE);
        trainSnackbar = Snackbar.make(container, "Training data...", Snackbar.LENGTH_INDEFINITE);

        View dialogView = getLayoutInflater().inflate(R.layout.dialog_edittext, null);
        EditText editText = dialogView.findViewById(R.id.edit_text);
        AlertDialog editDialog = new AlertDialog.Builder(MainActivity.this)
                .setTitle(R.string.enter_name)
                .setView(dialogView)
                .setPositiveButton(getString(R.string.ok), (dialogInterface, i) -> {
                    int idx = classifier.addPerson(editText.getText().toString());
                    performFileSearch(idx - 1);
                })
                .create();

        button = findViewById(R.id.add_button);
        button.setOnClickListener(view ->
                new AlertDialog.Builder(MainActivity.this)
                        .setTitle(getString(R.string.select_name))
                        .setItems(classifier.getClassNames(), (dialogInterface, i) -> {
                            if (i == 0) {
                                editDialog.show();
                            } else {
                                performFileSearch(i - 1);
                            }
                        })
                        .show());

        buttonSettings = findViewById(R.id.settings_button);
        buttonSettings.setOnClickListener(view ->
                startActivity(new Intent(MainActivity.this,SettingsActivity.class)));
    }

    @Override
    public void onResume() {
        super.onResume();
        checkPreferences();
    }

    private void checkPreferences() {
        SharedPreferences prefs = PreferenceManager.getDefaultSharedPreferences(getApplicationContext());
        reportOverSos = prefs.getBoolean("sendtosos",true);
        callsign = prefs.getString("callsign",null);
        if ((callsign == null) || (callsign.length() == 0)) {
            callsign = generateCallsign();
            SharedPreferences.Editor edit = prefs.edit();
            edit.putString("callsign",callsign).apply();
        }
        sensorId = callsign.replace(' ','-').toLowerCase()+"-afd";
        sosURL = prefs.getString("sosurl",null);
        sosUsername = prefs.getString("sosusr",null);
        sosPassword = prefs.getString("sospwd",null);
        sosConfidenceToReport = ((float)prefs.getInt("rptthreshold",65))/100f;
        maxReportingInterval = (long)prefs.getInt("maxrptrate",1)*1000l*60l;
    }

    private String generateCallsign() {
        Random random = new Random();
        StringWriter out = new StringWriter();

        out.append("AFD ");
        out.append(Integer.toString(random.nextInt(10)));
        out.append(Integer.toString(random.nextInt(10)));

        return out.toString();
    }

    @Override
    public void onDestroy() {
        if (sosService != null)
            sosService.shutdown();
        super.onDestroy();
    }

    @Override
    public void onPreviewSizeChosen(final Size size, final int rotation) {
        if (!initialized)
            new Thread(this::init).start();

        final float textSizePx =
        TypedValue.applyDimension(
                TypedValue.COMPLEX_UNIT_DIP, TEXT_SIZE_DIP, getResources().getDisplayMetrics());
        borderedText = new BorderedText(textSizePx);
        borderedText.setTypeface(Typeface.MONOSPACE);

        tracker = new MultiBoxTracker(this);

        previewWidth = size.getWidth();
        previewHeight = size.getHeight();

        sensorOrientation = rotation - getScreenOrientation();
        LOGGER.i("Camera orientation relative to screen canvas: %d", sensorOrientation);

        LOGGER.i("Initializing at size %dx%d", previewWidth, previewHeight);
        rgbFrameBitmap = Bitmap.createBitmap(previewWidth, previewHeight, Config.ARGB_8888);
        croppedBitmap = Bitmap.createBitmap(CROP_SIZE, CROP_SIZE, Config.ARGB_8888);

        frameToCropTransform =
                ImageUtils.getTransformationMatrix(
                        previewWidth, previewHeight,
                        CROP_SIZE, CROP_SIZE,
                        sensorOrientation, false);

        cropToFrameTransform = new Matrix();
        frameToCropTransform.invert(cropToFrameTransform);

        trackingOverlay = findViewById(R.id.tracking_overlay);
        trackingOverlay.addCallback(
                canvas -> {
                    tracker.draw(canvas);
                    if (isDebug()) {
                        tracker.drawDebug(canvas);
                    }
                });

        addCallback(
                canvas -> {
                    if (!isDebug()) {
                        return;
                    }
                    final Bitmap copy = cropCopyBitmap;
                    if (copy == null) {
                        return;
                    }

                    final int backgroundColor = Color.argb(100, 0, 0, 0);
                    canvas.drawColor(backgroundColor);

                    final Matrix matrix = new Matrix();
                    final float scaleFactor = 2;
                    matrix.postScale(scaleFactor, scaleFactor);
                    matrix.postTranslate(
                            canvas.getWidth() - copy.getWidth() * scaleFactor,
                            canvas.getHeight() - copy.getHeight() * scaleFactor);
                    canvas.drawBitmap(copy, matrix, new Paint());

                    final Vector<String> lines = new Vector<String>();
                    if (classifier != null) {
                        final String statString = classifier.getStatString();
                        final String[] statLines = statString.split("\n");
                        Collections.addAll(lines, statLines);
                    }
                    lines.add("");
                    lines.add("Frame: " + previewWidth + "x" + previewHeight);
                    lines.add("Crop: " + copy.getWidth() + "x" + copy.getHeight());
                    lines.add("View: " + canvas.getWidth() + "x" + canvas.getHeight());
                    lines.add("Rotation: " + sensorOrientation);
                    lines.add("Inference time: " + lastProcessingTimeMs + "ms");

                    borderedText.drawLines(canvas, 10, canvas.getHeight() - 10, lines);
                });
    }

    OverlayView trackingOverlay;

    void init() {
        runOnUiThread(()-> initSnackbar.show());
        File dir = new File(FileUtils.ROOT);

        if (!dir.isDirectory()) {
            if (dir.exists()) dir.delete();
            dir.mkdirs();

            AssetManager mgr = getAssets();
            FileUtils.copyAsset(mgr, FileUtils.DATA_FILE);
            FileUtils.copyAsset(mgr, FileUtils.MODEL_FILE);
            FileUtils.copyAsset(mgr, FileUtils.LABEL_FILE);
        }

        try {
            classifier = Classifier.getInstance(getAssets(), FACE_SIZE, FACE_SIZE);
        } catch (Exception e) {
            LOGGER.e("Exception initializing classifier!", e);
            finish();
        }

        runOnUiThread(()-> initSnackbar.dismiss());
        initialized = true;
    }

    @Override
    protected void processImage() {
        ++timestamp;
        final long currTimestamp = timestamp;
        byte[] originalLuminance = getLuminance();
        tracker.onFrame(
                previewWidth,
                previewHeight,
                getLuminanceStride(),
                sensorOrientation,
                originalLuminance,
                timestamp);
        trackingOverlay.postInvalidate();

        // No mutex needed as this method is not reentrant.
        if (computingDetection || !initialized || training) {
            readyForNextImage();
            return;
        }
        computingDetection = true;
        LOGGER.i("Preparing image " + currTimestamp + " for detection in bg thread.");

        rgbFrameBitmap.setPixels(getRgbBytes(), 0, previewWidth, 0, 0, previewWidth, previewHeight);

        if (luminanceCopy == null) {
            luminanceCopy = new byte[originalLuminance.length];
        }
        System.arraycopy(originalLuminance, 0, luminanceCopy, 0, originalLuminance.length);
        readyForNextImage();

        final Canvas canvas = new Canvas(croppedBitmap);
        canvas.drawBitmap(rgbFrameBitmap, frameToCropTransform, null);
        // For examining the actual TF input.
        if (SAVE_PREVIEW_BITMAP) {
            ImageUtils.saveBitmap(croppedBitmap);
        }

        runInBackground(
                () -> {
                    LOGGER.i("Running detection on image " + currTimestamp);
                    final long startTime = SystemClock.uptimeMillis();

                    cropCopyBitmap = Bitmap.createBitmap(croppedBitmap);
                    List<Classifier.Recognition> mappedRecognitions =
                            classifier.recognizeImage(croppedBitmap,cropToFrameTransform);

                    //check to see if there is something to report
                    if (reportOverSos && (sosURL != null) && (System.currentTimeMillis() > nextAvailableReportingTime)) {
                        Classifier.Recognition top = null;
                        if (mappedRecognitions != null) {
                            for (Classifier.Recognition recog : mappedRecognitions) {
                                if ((recog != null) && !"unknown".equalsIgnoreCase(recog.getTitle())) {
                                    if (top == null)
                                        top = recog;
                                    else {
                                        if (top.getConfidence() < recog.getConfidence())
                                            top = recog;
                                    }
                                }
                            }
                        }
                        //only report the highest confidence recognition
                        if ((top != null) && (top.getConfidence() > sosConfidenceToReport)) {
                            Log.d(TAG,"Reporting "+top.getTitle()+" ("+(int)(top.getConfidence()*100f)+"%) to SOS server");
                            if (sosSensor == null) {
                                sosSensor = new SosSensor(callsign, sensorId, "AFD", "Facial Recognition");
                                sensorMeasurementTime = new SensorMeasurementTime();
                                sensorMeasurementName = new SensorMeasurement(new SensorTextResultTemplateField("Name","http://www.sofwerx.org/afd.owl#name"));
                                sensorMeasurementConfidence = new SensorMeasurement(new SensorResultTemplateField("Confidence","http://www.sofwerx.org/afd.owl#confidence","p value"));
                                sosSensor.addMeasurement(sensorMeasurementTime);
                                sosSensor.addMeasurement(sensorMeasurementName);
                                sosSensor.addMeasurement(sensorMeasurementConfidence);
                            }
                            sensorMeasurementTime.setValue(System.currentTimeMillis());
                            sensorMeasurementName.setValue("\""+top.getTitle().replace(' ','-')+"\"");
                            sensorMeasurementConfidence.setValue(top.getConfidence());
                            if (sosService == null)
                                sosService = new SosService(MainActivity.this, sosSensor, sosURL, sosUsername, sosPassword, true, false);
                            else
                                sosService.broadcastSensorReadings();
                            nextAvailableReportingTime = System.currentTimeMillis() + maxReportingInterval;
                        }
                    }

                    lastProcessingTimeMs = SystemClock.uptimeMillis() - startTime;
                    tracker.trackResults(mappedRecognitions, luminanceCopy, currTimestamp);
                    trackingOverlay.postInvalidate();

                    requestRender();
                    computingDetection = false;
                });
    }

    @Override
    protected int getLayoutId() {
        return R.layout.camera_connection_fragment_tracking;
    }

    @Override
    protected Size getDesiredPreviewFrameSize() {
        return DESIRED_PREVIEW_SIZE;
    }

    @Override
    protected void onActivityResult(int requestCode, int resultCode, Intent data) {
        if (!initialized) {
            Snackbar.make(
                    getWindow().getDecorView().findViewById(R.id.container),
                    "Try it again later", Snackbar.LENGTH_SHORT)
                    .show();
            return;
        }

        if (resultCode == RESULT_OK) {
            trainSnackbar.show();
            button.setEnabled(false);
            training = true;

            ClipData clipData = data.getClipData();
            ArrayList<Uri> uris = new ArrayList<>();

            if (clipData == null) {
                uris.add(data.getData());
            } else {
                for (int i = 0; i < clipData.getItemCount(); i++)
                    uris.add(clipData.getItemAt(i).getUri());
            }

            new Thread(() -> {
                try {
                    classifier.updateData(requestCode, getContentResolver(), uris);
                } catch (Exception e) {
                    LOGGER.e(e, "Exception!");
                } finally {
                    training = false;
                }
                runOnUiThread(() -> {
                    trainSnackbar.dismiss();
                    button.setEnabled(true);
                });
            }).start();

        }
    }

    public void performFileSearch(int requestCode) {
        Intent intent = new Intent(Intent.ACTION_OPEN_DOCUMENT);
        intent.addCategory(Intent.CATEGORY_OPENABLE);
        intent.putExtra(Intent.EXTRA_ALLOW_MULTIPLE, true);
        intent.setType("image/*");

        startActivityForResult(intent, requestCode);
    }
}
