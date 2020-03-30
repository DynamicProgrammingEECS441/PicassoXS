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
        
       
// //       let parameters = ["name": rname] //Optional for extra parameter
        
//         let headers: HTTPHeaders = ["content-type": "application/json"]
//         let jsonObject: [String: Any] = [
//                  "signature_name": "predict_images",
//                  "instances": imgData
//             ]
//         let jsonData = try? JSONSerialization.data(withJSONObject: jsonObject)
        
        
        
//         AF.upload(jsonData!,to: "http://35.239.199.254:8501/v1/models/model:predict", headers: headers).responseJSON { response in
//             debugPrint(response)
//         }
        


        // first
        // let url:NSURL = NSURL(string: url_to_request)!
        // let session = NSURLSession.sharedSession()
        
        // let request = NSMutableURLRequest(URL: url)
        // request.HTTPMethod = "POST"
        // request.cachePolicy = NSURLRequestCachePolicy.ReloadIgnoringCacheData
    
        
        // //let data = "data=Hi".dataUsingEncoding(NSUTF8StringEncoding)
        // let data = imgData
        
        // let task = session.uploadTaskWithRequest(request, fromData: data, completionHandler:
        //     {(data,response,error) in
            
        //         guard let _:NSData = data, let _:NSURLResponse = response  where error == nil else {
        //             print("error")
        //             return
        //         }
                
        //         let dataString = NSString(data: data!, encoding: NSUTF8StringEncoding)
        //         print(dataString)
        //     }
        // );
        
        // task.resume()
     

        // second
        // guard let png = UIImagePNGRepresentation(image) else{
        //     print("error")
        //     return
        // }

        // //Set up a network request
        // let request = NSMutableURLRequest()
        // request.HTTPMethod = "POST"
        // request.URL = NSURL(string: "http://127.0.0.1:5000/")
        // request.setValue("application/x-www-form-urlencoded", forHTTPHeaderField: "Content-Type")
        // request.setValue("\(png.length)", forHTTPHeaderField: "Content-Length")
        // request.HTTPBody = png
        // // Figure out what the request is making and the encoding type...

        // //Execute the network request
        // let upload = NSURLSession.sharedSession().uploadTaskWithRequest(request, fromData: png) { (data: NSData?, response: NSURLResponse?, error: NSError?) -> Void in
        //     //What you want to do when the upload succeeds or fails
        // }

        // upload.resume()
        


        let url = URL(string: "............")

        // generate boundary string using a unique per-app string
        let boundary = UUID().uuidString

        let session = URLSession.shared

        // Set the URLRequest to POST and to the specified URL
        var urlRequest = URLRequest(url: url!)
        urlRequest.httpMethod = "POST"

        // Set Content-Type Header to multipart/form-data, this is equivalent to submitting form data with file upload in a web browser
        // And the boundary is also set here
        urlRequest.setValue("multipart/form-data; boundary=\(boundary)", forHTTPHeaderField: "Content-Type")

        var data = Data()

        // Add the image data to the raw http request data
        data.append("\r\n--\(boundary)\r\n".data(using: .utf8)!)
        data.append("Content-Disposition: form-data; name=\"\(paramName)\"; filename=\"\(fileName)\"\r\n".data(using: .utf8)!)
        data.append("Content-Type: image/png\r\n\r\n".data(using: .utf8)!)
        data.append(imgData!)

        data.append("\r\n--\(boundary)--\r\n".data(using: .utf8)!)

        // Send a POST request to the URL, with the data we created earlier
        session.uploadTask(with: urlRequest, from: data, completionHandler: { responseData, response, error in
            if error == nil {
                let jsonData = try? JSONSerialization.jsonObject(with: responseData!, options: .allowFragments)
                if let json = jsonData as? [String: Any] {
                    print(json)
                }
            }
        }).resume()
    
    }
    
    
}




extension UIImage {
    var jpeg: Data? { jpegData(compressionQuality: 0.8) }  // QUALITY min = 0 / max = 1
    var png: Data? { pngData() }
}
