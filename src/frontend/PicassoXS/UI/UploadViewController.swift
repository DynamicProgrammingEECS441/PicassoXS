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



class UploadViewController: UIViewController{
    
    
    @IBOutlet weak var InputIm: UIImageView!
    @IBOutlet weak var FilterIm: UIImageView!

    override func viewDidLoad() {
        super.viewDidLoad()
        InputIm.image = GlobalVariables.curInput
        FilterIm.image = GlobalVariables.curFilter
        sendRequest(Image: InputIm.image!)
    }
    
    //Upload Filter Image for Filtering
    //TODO: IMPLEMENT SELECTION FOR DIFFERENT FILTERS
    func sendRequest(Image: UIImage){
//        "http://35.239.199.254:8501/v1/models/model:predict")!)
//        var request = URLRequest(url: URL(string: "http://0.0.0.0:8000/")!)
        
        let imgData = Image.jpegData(compressionQuality: 0.1)!
        
        
//       let parameters = ["name": rname] //Optional for extra parameter
        
        let headers: HTTPHeaders = ["content-type": "application/json"]
        let jsonObject: [String: Any] = [
                 "signature_name": "predict_images",
                 "instances": imgData
            ]
        let jsonData = try? JSONSerialization.data(withJSONObject: jsonObject)
        
        
        
        AF.upload(jsonData!,to: "http://35.239.199.254:8501/v1/models/model:predict", headers: headers).responseJSON { response in
            debugPrint(response)
        }
        
     
        
    
    }
    
    
}




extension UIImage {
    var jpeg: Data? { jpegData(compressionQuality: 0.8) }  // QUALITY min = 0 / max = 1
    var png: Data? { pngData() }
}
