// Tencent is pleased to support the open source community by making ncnn available.
//
// Copyright (C) 2025 THL A29 Limited, a Tencent company. All rights reserved.
//
// Licensed under the BSD 3-Clause License (the "License"); you may not use this file except
// in compliance with the License. You may obtain a copy of the License at
//
// https://opensource.org/licenses/BSD-3-Clause
//
// Unless required by applicable law or agreed to in writing, software distributed
// under the License is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR
// CONDITIONS OF ANY KIND, either express or implied. See the License for the
// specific language governing permissions and limitations under the License.

package io.github.yusufyorunc.objectdetector;

import android.Manifest;
import android.content.pm.PackageManager;
import android.graphics.PixelFormat;
import android.os.Bundle;
import android.util.Log;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.WindowManager;
import android.widget.AdapterView;
import android.widget.Button;
import android.widget.Spinner;
import android.widget.Toast;

import androidx.activity.result.ActivityResultLauncher;
import androidx.activity.result.contract.ActivityResultContracts;
import androidx.annotation.NonNull;
import androidx.appcompat.app.AppCompatActivity;
import androidx.core.app.ActivityCompat;
import androidx.core.content.ContextCompat;

import java.util.function.IntConsumer;

public class MainActivity extends AppCompatActivity implements SurfaceHolder.Callback
{
    private static final String TAG = "MainActivity";
    private static final String CAMERA_PERMISSION = Manifest.permission.CAMERA;
    private static final int CAMERA_BACK = 0;
    private static final int CAMERA_FRONT = 1;

    private final YOLO11Ncnn yolo11ncnn = new YOLO11Ncnn();

    private int facing = CAMERA_BACK;
    private int currentTask = 0;
    private int currentModel = 0;
    private int currentCpugpu = 0;

    private boolean isSurfaceReady = false;
    private boolean isCameraOpened = false;

    private SurfaceView cameraView;

    private final ActivityResultLauncher<String> cameraPermissionLauncher =
            registerForActivityResult(new ActivityResultContracts.RequestPermission(), isGranted -> {
                if (isGranted)
                {
                    startCameraIfReady();
                }
                else
                {
                    Toast.makeText(this, R.string.camera_permission_required, Toast.LENGTH_LONG).show();
                }
            });

    @Override
    protected void onCreate(Bundle savedInstanceState)
    {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.main);

        getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);

        setupCameraView();
        setupControls();
        reloadModel();
    }

    private void setupCameraView()
    {
        cameraView = findViewById(R.id.cameraview);

        final SurfaceHolder holder = cameraView.getHolder();
        holder.setFormat(PixelFormat.RGBA_8888);
        holder.addCallback(this);
    }

    private void setupControls()
    {
        final Button buttonSwitchCamera = findViewById(R.id.buttonSwitchCamera);
        buttonSwitchCamera.setOnClickListener(v -> switchCamera());

        bindSpinner(findViewById(R.id.spinnerTask), value -> currentTask = value);
        bindSpinner(findViewById(R.id.spinnerModel), value -> currentModel = value);
        bindSpinner(findViewById(R.id.spinnerCPUGPU), value -> currentCpugpu = value);
    }

    private void bindSpinner(Spinner spinner, IntConsumer onChanged)
    {
        final int[] lastValue = {spinner.getSelectedItemPosition()};
        onChanged.accept(lastValue[0]);

        spinner.setOnItemSelectedListener(new AdapterView.OnItemSelectedListener() {
            @Override
            public void onItemSelected(AdapterView<?> parent, View view, int position, long id)
            {
                if (position == lastValue[0])
                {
                    return;
                }

                lastValue[0] = position;
                onChanged.accept(position);
                reloadModel();
            }

            @Override
            public void onNothingSelected(AdapterView<?> parent)
            {
            }
        });
    }

    private void switchCamera()
    {
        facing = (facing == CAMERA_BACK) ? CAMERA_FRONT : CAMERA_BACK;
        stopCameraIfOpened();
        startCameraIfReady();
    }

    private boolean hasCameraPermission()
    {
        return ContextCompat.checkSelfPermission(this, CAMERA_PERMISSION) == PackageManager.PERMISSION_GRANTED;
    }

    private void requestCameraPermission()
    {
        if (ActivityCompat.shouldShowRequestPermissionRationale(this, CAMERA_PERMISSION))
        {
            Toast.makeText(this, R.string.camera_permission_rationale, Toast.LENGTH_SHORT).show();
        }

        cameraPermissionLauncher.launch(CAMERA_PERMISSION);
    }

    private void startCameraIfReady()
    {
        if (!hasCameraPermission() || !isSurfaceReady || isCameraOpened)
        {
            return;
        }

        isCameraOpened = yolo11ncnn.openCamera(facing);
        if (!isCameraOpened)
        {
            Log.e(TAG, "openCamera failed");
        }
    }

    private void stopCameraIfOpened()
    {
        if (!isCameraOpened)
        {
            return;
        }

        if (!yolo11ncnn.closeCamera())
        {
            Log.e(TAG, "closeCamera failed");
        }

        isCameraOpened = false;
    }

    private void updateOutputWindow(SurfaceHolder holder)
    {
        if (!yolo11ncnn.setOutputWindow(holder.getSurface()))
        {
            Log.e(TAG, "setOutputWindow failed");
        }
    }

    private void reloadModel()
    {
        final boolean loaded = yolo11ncnn.loadModel(getAssets(), currentTask, currentModel, currentCpugpu);
        if (!loaded)
        {
            Log.e(TAG, "loadModel failed");
        }
    }

    @Override
    public void surfaceCreated(@NonNull SurfaceHolder holder)
    {
        isSurfaceReady = true;
        updateOutputWindow(holder);
        startCameraIfReady();
    }

    @Override
    public void surfaceChanged(@NonNull SurfaceHolder holder, int format, int width, int height)
    {
        updateOutputWindow(holder);
    }

    @Override
    public void surfaceDestroyed(@NonNull SurfaceHolder holder)
    {
        isSurfaceReady = false;
        stopCameraIfOpened();
    }

    @Override
    protected void onResume()
    {
        super.onResume();

        if (!hasCameraPermission())
        {
            requestCameraPermission();
            return;
        }

        startCameraIfReady();
    }

    @Override
    protected void onPause()
    {
        stopCameraIfOpened();
        super.onPause();
    }

    @Override
    protected void onDestroy()
    {
        if (cameraView != null)
        {
            cameraView.getHolder().removeCallback(this);
        }

        super.onDestroy();
    }
}
