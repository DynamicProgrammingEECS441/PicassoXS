//
//  UploadViewController.swift
//  PicassoXs
//
//  Created by Senhao Wang on 3/23/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import Foundation
import UIKit
import Alamofire
import AlamofireImage


public struct imageJson : Codable {
   public let signature_name:String
   public let instances:String
}


class UploadViewController: UIViewController{
    
    
    @IBOutlet weak var InputIm: UIImageView!
    @IBOutlet weak var FilterIm: UIImageView!
    @IBOutlet weak var OutputIm: UIImageView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        InputIm.image = GlobalVariables.curInput
        FilterIm.image = GlobalVariables.curFilter
        //OutputIm.setGIFImage(name: "Loading")
        
        if(GlobalVariables.UploadMethod != 0){
            sendRequest(Image: InputIm.image!)
        }
        else{
            sendRequestWithStyle(Image: InputIm.image!, ImageStyle: FilterIm.image!)
        }
        
    }
    
    override func viewDidDisappear(_ animated: Bool) {
        super.viewDidDisappear(animated)
        OutputIm.stopAnimating()
    }
    
    func sendRequest(Image: UIImage){
        let imgData = Image.jpegData(compressionQuality: 1)!
            AF.upload(multipartFormData: { multipartFormData in
                multipartFormData.append(imgData, withName: GlobalVariables.codeRef[GlobalVariables.UploadMethod], fileName: "document.png", mimeType: "image/png")
    //        }, to: "http://localhost:8000/general_model_grpc/")
            }, to: "https://tensorflow-serving-9905.appspot.com/general_model_grpc/")
                .responseImage { response in
                    if case .success(let image) = response.result {
                        print("image downloaded: \(image)")
                        self.performSegue(withIdentifier: "success", sender: nil)
                        GlobalVariables.curResult = image
                    }
                }
        }
    
    
    func sendRequestWithStyle(Image: UIImage, ImageStyle: UIImage){
        let imgData = Image.jpegData(compressionQuality: 1)!
        let imgStyleData = ImageStyle.jpegData(compressionQuality: 1)!
            
            AF.upload(multipartFormData: { multipartFormData in
                multipartFormData.append(imgData, withName: "content_img", fileName: "content.png", mimeType: "image/png")
                multipartFormData.append(imgStyleData, withName: "style_img", fileName: "style.png", mimeType: "image/png")
    //        }, to: "http://localhost:8000/arbitrary_style_grpc/")
            }, to: "https://tensorflow-serving-9905.appspot.com/arbitrary_style_grpc/")
                .responseImage { response in
                    if case .success(let image) = response.result {
                        print("image downloaded: \(image)")
                        
                        self.performSegue(withIdentifier: "success", sender: nil)
                        GlobalVariables.curResult = image
            }
        }
    }
    
    
    
    
    
    
    
}
    

extension UIImage {
    var jpeg: Data? { jpegData(compressionQuality: 0.8) }  // QUALITY min = 0 / max = 1
    var png: Data? { pngData() }
    
    func resized(MAX_PIX: CGFloat) -> UIImage {
        
        var width = self.size.width
        var height = self.size.height
        
        if(width >= height){
            let ratio = MAX_PIX/width
            height = height * ratio
            width = MAX_PIX
        }
        else{
            let ratio = MAX_PIX/height
            height = MAX_PIX
            width = width * ratio
        }
        
        return UIGraphicsImageRenderer(size: size).image { _ in
            draw(in: CGRect(origin: .zero, size: CGSize(width: width, height: height)))
        }
    }
    
    
    
    class func makeGIFFromCollection(name: String, repeatCount: Int = 0) -> GIF? {
        guard let path = Bundle.main.path(forResource: name, ofType: "gif") else {
            print("Cannot find a path from the file \"\(name)\"")
            return nil
        }

        let url = URL(fileURLWithPath: path)
        let data = try? Data(contentsOf: url)
        guard let d = data else {
            print("Cannot turn image named \"\(name)\" into data")
            return nil
        }

        return makeGIFFromData(data: d, repeatCount: repeatCount)
    }

    class func makeGIFFromData(data: Data, repeatCount: Int = 0) -> GIF? {
        guard let source = CGImageSourceCreateWithData(data as CFData, nil) else {
            print("Source for the image does not exist")
            return nil
        }

        let count = CGImageSourceGetCount(source)
        var images = [UIImage]()
        var duration = 0.0

        for i in 0..<count {
            if let cgImage = CGImageSourceCreateImageAtIndex(source, i, nil) {
                let image = UIImage(cgImage: cgImage)
                images.append(image)

                let delaySeconds = UIImage.delayForImageAtIndex(Int(i),
                                                                source: source)
                duration += delaySeconds
            }
        }

        return GIF(images: images, durationInSec: duration, repeatCount: repeatCount)
    }

    class func delayForImageAtIndex(_ index: Int, source: CGImageSource!) -> Double {
        var delay = 0.0

        // Get dictionaries
        let cfProperties = CGImageSourceCopyPropertiesAtIndex(source, index, nil)
        let gifPropertiesPointer = UnsafeMutablePointer<UnsafeRawPointer?>.allocate(capacity: 0)
        if CFDictionaryGetValueIfPresent(cfProperties, Unmanaged.passUnretained(kCGImagePropertyGIFDictionary).toOpaque(), gifPropertiesPointer) == false {
            return delay
        }

        let gifProperties:CFDictionary = unsafeBitCast(gifPropertiesPointer.pointee, to: CFDictionary.self)

        // Get delay time
        var delayObject: AnyObject = unsafeBitCast(
            CFDictionaryGetValue(gifProperties,
                                 Unmanaged.passUnretained(kCGImagePropertyGIFUnclampedDelayTime).toOpaque()),
            to: AnyObject.self)
        if delayObject.doubleValue == 0 {
            delayObject = unsafeBitCast(CFDictionaryGetValue(gifProperties,
                                                             Unmanaged.passUnretained(kCGImagePropertyGIFDelayTime).toOpaque()), to: AnyObject.self)
        }

        delay = delayObject as? Double ?? 0

        return delay
    }
}



extension UIImageView {
    func setGIFImage(name: String, repeatCount: Int = 0 ) {
        DispatchQueue.global().async {
            if let gif = UIImage.makeGIFFromCollection(name: name, repeatCount: repeatCount) {
                DispatchQueue.main.async {
                    self.setImage(withGIF: gif)
                    self.startAnimating()
                }
            }
        }
    }

    private func setImage(withGIF gif: GIF) {
        animationImages = gif.images
        animationDuration = gif.durationInSec
        animationRepeatCount = gif.repeatCount
    }
}


class GIF: NSObject {
    let images: [UIImage]
    let durationInSec: TimeInterval
    let repeatCount: Int

    init(images: [UIImage], durationInSec: TimeInterval, repeatCount: Int = 0) {
        self.images = images
        self.durationInSec = durationInSec
        self.repeatCount = repeatCount
    }
}
