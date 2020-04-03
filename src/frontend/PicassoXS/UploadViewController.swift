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



class UploadViewController: UIViewController{
    
    
    @IBOutlet weak var InputIm: UIImageView!
    @IBOutlet weak var FilterIm: UIImageView!
    @IBOutlet weak var OutputIm: UIImageView!

    override func viewDidLoad() {
        super.viewDidLoad()
        InputIm.image = GlobalVariables.curInput
        FilterIm.image = GlobalVariables.curFilter
        sendRequest(Image: InputIm.image!)
    }
    
    //Upload Filter Image for Filtering
    //TODO: IMPLEMENT SELECTION FOR DIFFERENT FILTERS
    func sendRequest(Image: UIImage){
      
        let imgData = Image.jpegData(compressionQuality: 1)!
        
        AF.upload(multipartFormData: { multipartFormData in
            multipartFormData.append(imgData, withName: "document", fileName: "document.png", mimeType: "image/png")
        }, to: "http://localhost:8000/upload_img/")
            .responseImage { response in
                if case .success(let image) = response.result {
                    print("image downloaded: \(image)")
                    self.OutputIm.image = image
                }
            }
    }
}




extension UIImage {
    var jpeg: Data? { jpegData(compressionQuality: 1) }  // QUALITY min = 0 / max = 1
    var png: Data? { pngData() }
}
