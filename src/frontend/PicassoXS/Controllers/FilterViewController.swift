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

class FilterViewController : UIViewController, UIImagePickerControllerDelegate, UINavigationControllerDelegate, UICollectionViewDelegate, UICollectionViewDataSource{
    
    
    func collectionView(_ collectionView: UICollectionView, numberOfItemsInSection section: Int) -> Int {
        return GlobalVariables.Filters.count
    }

    func collectionView(_ collectionView: UICollectionView, cellForItemAt indexPath: IndexPath) -> UICollectionViewCell {
        print("Counting!!!   ", indexPath.item)
        let filter = GlobalVariables.Filters[indexPath.item]
        let cell = collectionView.dequeueReusableCell(withReuseIdentifier: "FilterViewCell", for: indexPath) as! FilterViewCell
        cell.setView(filter: filter)
        
        return cell
    }
    
    
    
    func collectionView(_ collectionView: UICollectionView, didSelectItemAt indexPath: IndexPath) {
        DetailIndex = indexPath.item
        GlobalVariables.UploadMethod = indexPath.item
        if (DetailIndex > 0){
            performSegue(withIdentifier: "Detail", sender: self)
        }
        else if(DetailIndex == 0){
            PickImage()
        }
    }
    
    
    
    
//    func tableView(_ tableView: UITableView, numberOfRowsInSection section: Int) -> Int {
//        return GlobalVariables.Filters.count
//    }
//
//    func tableView(_ tableView: UITableView, cellForRowAt indexPath: IndexPath) -> UITableViewCell {
//        let filter = GlobalVariables.Filters[indexPath.row]
//        let cell = tableView.dequeueReusableCell(withIdentifier: "FilterViewCell") as! FilterViewCell
//        cell.setFilter(filter: filter)
//        return cell
//    }
    
//    func tableView(_ tableView: UITableView, didSelectRowAt indexPath: IndexPath) {
//        DetailIndex = indexPath.row
//        GlobalVariables.UploadMethod = indexPath.row
//        if (DetailIndex > 0){
//            performSegue(withIdentifier: "Detail", sender: self)
//        }
//        else if(DetailIndex == 0){
//            PickImage()
//        }
//    }
//
    
    
    override func viewDidLoad() {
        
        super.viewDidLoad()
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
    
    //用户取消选择
     func imagePickerControllerDidCancel(_ picker: UIImagePickerController) {
         dismiss(animated: true, completion: nil)
     }
    
    //用户选择完成
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
