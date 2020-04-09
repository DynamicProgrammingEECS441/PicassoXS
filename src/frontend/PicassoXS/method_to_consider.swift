/// Method one to allow portrait Mode

import UIKit

class ViewController: UIViewController {

    override func viewDidLoad() {
        super.viewDidLoad()
        NotificationCenter.default.addObserver(self, selector: #selector(MainCameraVC.orientationChanged(notification:)), name: Notification.Name.UIDeviceOrientationDidChange, object: nil)
    }

    func orientationChanged(notification: Notification) {
        let deviceOrientation = UIDevice.current.orientation
        var angle: Double?

        switch deviceOrientation {
        case .portrait:
            angle = 0
            break
        case .portraitUpsideDown:
            angle = Double.pi
            break
        case .landscapeLeft:
            angle = Double.pi / 2
            break
        case .landscapeRight:
            angle = -Double.pi / 2
            break
        default:
            break
        }

        if let angle = angle {
            let transform = CGAffineTransform(rotationAngle: CGFloat(angle))

            UIView.animate(withDuration: 0.3, animations: {
                self.label.transform = transform
                self.button.transform = transform
                ...
            })
        }
    }
}

/// Method II to enable portrait mode 

// private let photoOutput = AVCapturePhotoOutput()
...
// Check that portrait effects matte delivery is supported on this particular device:
if self.photoOutput.isPortraitEffectsMatteDeliverySupported {
    self.photoOutput.isPortraitEffectsMatteDeliveryEnabled = true
}