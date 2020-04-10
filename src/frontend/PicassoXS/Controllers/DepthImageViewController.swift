//
//  DepthImageViewController.swift
//  PicassoXFinal
//
//  Created by Senhao Wang on 4/10/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import UIKit
import AVFoundation

struct Portrait{
    static var depthCI : CIImage?
    static var depthUI : UIImage?
    static var RGB: UIImage?
}


class RecorderViewController: UIViewController {

    @IBOutlet weak var previewView: UIView!
    
    @IBOutlet weak var button: UIButton!
    @IBAction func onTapTakePhoto(_ sender: Any) {

        guard var capturePhotoOutput = self.capturePhotoOutput else { return }

        var photoSettings = AVCapturePhotoSettings()
        photoSettings.isDepthDataDeliveryEnabled = true
        
        capturePhotoOutput.capturePhoto(with: photoSettings, delegate: self)

    }

    var session: AVCaptureSession?
    var videoPreviewLayer: AVCaptureVideoPreviewLayer?
    var capturePhotoOutput: AVCapturePhotoOutput?


    override func viewDidLoad() {
        super.viewDidLoad()

        AVCaptureDevice.requestAccess(for: .video, completionHandler: { _ in })

        let captureDevice = AVCaptureDevice.default(.builtInDualCamera, for: .video, position: .back)

        print(captureDevice!.activeDepthDataFormat)

        do{
            let input = try AVCaptureDeviceInput(device: captureDevice!)

            self.capturePhotoOutput = AVCapturePhotoOutput()

            self.session = AVCaptureSession()
            self.session?.beginConfiguration()
            self.session?.sessionPreset = .photo
            self.session?.addInput(input)

            self.videoPreviewLayer = AVCaptureVideoPreviewLayer(session: self.session!)
            self.videoPreviewLayer?.videoGravity = AVLayerVideoGravity.resizeAspectFill
            self.videoPreviewLayer?.frame = self.view.layer.bounds
            self.previewView.layer.addSublayer(self.videoPreviewLayer!)
            view.sendSubviewToBack(self.previewView)
            
            self.session?.addOutput(self.capturePhotoOutput!)
            self.session?.commitConfiguration()
            self.capturePhotoOutput?.isDepthDataDeliveryEnabled = true
            self.session?.startRunning()
        }
        catch{
            print(error)
        }
        view.bringSubviewToFront(button)

    }

    override func didReceiveMemoryWarning() {
        super.didReceiveMemoryWarning()
        // Dispose of any resources that can be recreated.
    }

}

extension RecorderViewController : AVCapturePhotoCaptureDelegate {

    func photoOutput(_ output: AVCapturePhotoOutput, didFinishProcessingPhoto photo: AVCapturePhoto, error: Error?) {
        
        Portrait.depthCI = CIImage(depthData: photo.depthData!)
        Portrait.RGB = UIImage(data: photo.fileDataRepresentation()!)
        GlobalVariables.curInput = Portrait.RGB
        
        Portrait.depthUI =  UIImage(ciImage: CIImage(depthData: photo.depthData!)!.oriented(forExifOrientation: Int32(CGImagePropertyOrientation.right.rawValue)))
        
        self.performSegue(withIdentifier: "ViewDepth", sender: nil)
    }
    
    


}
