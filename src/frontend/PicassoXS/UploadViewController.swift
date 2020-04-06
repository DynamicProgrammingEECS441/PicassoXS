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
//            sendRequest(Image: InputIm.image!)
            sendRequestWithStyle(Image: InputIm.image!, ImageStyle: FilterIm.image!)
        }
    
    //Upload Filter Image for Filtering
    //TODO: IMPLEMENT SELECTION FOR DIFFERENT FILTERS
    func sendRequest(Image: UIImage){
        let imgData = Image.jpegData(compressionQuality: 1)!
        AF.upload(multipartFormData: { multipartFormData in
            multipartFormData.append(imgData, withName: "document", fileName: "document.png", mimeType: "image/png")
        }, to: "http://localhost:8000/general_model_grpc/")
//        }, to: "https://tensorflow-serving-9905.appspot.com/upload_img/")
            .responseImage { response in
                if case .success(let image) = response.result {
                    print("image downloaded: \(image)")
                    self.OutputIm.image = image
                }
            }
    }
    
    func sendRequestWithStyle(Image: UIImage, ImageStyle: UIImage){
        let imgData = Image.jpegData(compressionQuality: 1)!
        let imgStyleData = ImageStyle.jpegData(compressionQuality: 1)!
        
        AF.upload(multipartFormData: { multipartFormData in
            multipartFormData.append(imgData, withName: "content_img", fileName: "content.png", mimeType: "image/png")
            multipartFormData.append(imgStyleData, withName: "style_img", fileName: "style.png", mimeType: "image/png")
        }, to: "http://localhost:8000/arbitrary_style_grpc/")
//        }, to: "https://tensorflow-serving-9905.appspot.com/upload_img/")
            .responseImage { response in
                if case .success(let image) = response.result {
                    print("image downloaded: \(image)")
                    self.OutputIm.image = image
                }
            }
    }
    
}
    

extension UIImage {
    var jpeg: Data? { jpegData(compressionQuality: 0.8) }  // QUALITY min = 0 / max = 1
    var png: Data? { pngData() }
}