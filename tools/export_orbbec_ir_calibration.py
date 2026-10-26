"""Read connected Orbbec IR calibration without changing device settings."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--serial', default='CPC93630004R')
    parser.add_argument('--width', type=int, default=1280)
    parser.add_argument('--height', type=int, default=800)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('Output already exists; choose a new file')
    try:
        from pyorbbecsdk import Context, Pipeline, OBSensorType, OBFormat, OBPropertyID
    except ImportError as error:
        raise SystemExit('Install a compatible official pyorbbecsdk v2 package first: ' + str(error))
    devices = Context().query_devices()
    device = None
    for index in range(devices.get_count()):
        candidate = devices.get_device_by_index(index)
        if candidate.get_device_info().get_serial_number() == args.serial:
            device = candidate
            break
    if device is None:
        raise SystemExit('Requested camera not connected: ' + args.serial)
    info = device.get_device_info()
    result = {'serial': info.get_serial_number(), 'width': args.width, 'height': args.height,
              'historical_capture_state_verified': False, 'errors': {}}
    for label, method in [('name', 'get_name'), ('firmware', 'get_firmware_version')]:
        try:
            result[label] = getattr(info, method)()
        except Exception as error:
            result['errors'][label] = str(error)
    try:
        result['current_ir_rectify'] = device.get_bool_property(OBPropertyID.OB_PROP_IR_RECTIFY_BOOL)
    except Exception as error:
        result['current_ir_rectify'] = None
        result['errors']['current_ir_rectify'] = str(error)
    pipeline = Pipeline(device)
    profiles = {}
    for side, sensor in [('left', OBSensorType.LEFT_IR_SENSOR), ('right', OBSensorType.RIGHT_IR_SENSOR)]:
        try:
            profile = pipeline.get_stream_profile_list(sensor).get_video_stream_profile(args.width, args.height, OBFormat.Y8, 0)
            profiles[side] = profile
            intr = profile.get_intrinsic()
            dist = profile.get_distortion()
            result[side] = {
                'intrinsic': {k: getattr(intr, k) for k in ['fx','fy','cx','cy','width','height']},
                'distortion': {k: float(getattr(dist, k)) for k in ['k1','k2','k3','k4','k5','k6','p1','p2']},
                'distortion_model': str(getattr(dist, 'model', 'unknown')),
                'fps': profile.get_fps(),
            }
        except Exception as error:
            result['errors'][side] = str(error)
    if len(profiles) == 2:
        try:
            ext = profiles['left'].get_extrinsic_to(profiles['right'])
            result['left_to_right'] = {'rotation_row_major': [float(v) for v in ext.rot],
                                       'translation_mm': [float(v) for v in ext.transform]}
        except Exception as error:
            result['errors']['left_to_right'] = str(error)
    result['export_complete'] = all(k in result for k in ['left', 'right', 'left_to_right'])
    result['note'] = 'SDK stream-profile parameters; verify correspondence to original unrectified captures before rectifying. Current state does not prove past capture state.'
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result['export_complete']:
        raise SystemExit('Incomplete export: preserve the JSON errors for diagnosis; do not substitute D2C parameters.')


if __name__ == '__main__':
    main()
