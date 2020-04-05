//
//  CameraViewController.swift
//  PicassoXs
//
//  Created by Senhao Wang on 3/22/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import Foundation
import UIKit

var DetailIndex=0

class FilterViewController : UIViewController, UITableViewDelegate, UITableViewDataSource, UIImagePickerControllerDelegate, UINavigationControllerDelegate{
    
    
    
    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
        return GlobalVariables.Filters.count
    }
    
    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
        let filter = GlobalVariables.Filters[indexPath.row]
        let cell = tableView.dequeueReusableCell(withIdentifier: "TableViewCell") as! TableViewCell
        cell.setFilter(filter: filter)
        return cell
    }
    
    
    @IBOutlet weak var FilterTableView: UITableView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
    }
    
    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
        DetailIndex = indexPath.row
        if (DetailIndex > 0){
            performSegue(withIdentifier: "Detail", sender: self)
        }
        else if(DetailIndex == 0){
            PickImage()
        }
        
    }
    
    func PickImage(){
        print("Library Tapped")
        if UIImagePickerController.isSourceTypeAvailable(UIImagePickerController.SourceType.photoLibrary) {
          let imagePicker = UIImagePickerController()
          imagePicker.delegate = self
            imagePicker.sourceType = UIImagePickerController.SourceType.photoLibrary
          imagePicker.allowsEditing = false
          self.present(imagePicker, animated: true, completion: nil)
        }
    }
    
    //用户取消拍照
     func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
         dismiss(animated: true, completion: nil)
     }
    
    //用户拍照完成
     func imagePickerController(_ picker: UIImagePickerController, didFinishPickingMediaWithInfo info: [UIImagePickerController.InfoKey : Any]) {
         print("\(info)")
         if let image = info[.originalImage] as? UIImage {
            
            //Set Current Input Image
            GlobalVariables.curFilter=image
            dismiss(animated: true, completion: nil)
            performSegue(withIdentifier: "UploadOwn", sender: nil)
         }
     }
    
    
}
