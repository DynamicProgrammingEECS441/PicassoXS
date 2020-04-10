//
//  PortraitAdjustViewController.swift
//  PicassoXFinal
//
//  Created by Senhao Wang on 4/10/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import Foundation
import UIKit

class PortraitAdjustViewController: UIViewController{
    
    @IBOutlet weak var Result: UIImageView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        Result.image = UIImage(cgImage: (GlobalVariables.curResult?.cgImage)!, scale: 1.0, orientation: .right)
        
        
    }
    
    @IBOutlet weak var slider: UISlider!
    
    @IBAction func depthSlider(_ sender: Any) {
        print(slider.value)
        
        let ciImage = CIImage(image: GlobalVariables.curResult!)
         
         Result.image = blur(image: ciImage!, mask: Portrait.depthCI!, orientation: .right, value: slider.value)
          
    }

    
    
    @IBOutlet weak var Save: UIButton!
    
    @IBAction func SaveToAlbum(_ sender: Any) {
        UIImageWriteToSavedPhotosAlbum(self.Result.image!, nil, nil, nil);
        Save.tintColor = GlobalVariables.Grey1
    }
    
    
    
    @IBAction func Continue(_ sender: Any) {
        self.performSegue(withIdentifier: "Home2", sender: nil)
        GlobalVariables.FilteredHistory.insert(MadeImages(GlobalVariables.curInput!, GlobalVariables.curFilter!, Result.image!), at: 0)
    }
    
    
     
    func blur(image: CIImage, mask: CIImage, orientation: UIImage.Orientation = .up, value: Float) -> UIImage? {
        
        let context = CIContext()

       // 2
       let output = image.applyingFilter("CIBlendWithMask", parameters: ["inputMaskImage" : mask])

       // 3
       guard let cgImage = context.createCGImage(output, from: output.extent) else {
         return nil
       }

       // 4
       return UIImage(cgImage: cgImage, scale: 1.0, orientation: orientation)
     }

     
}
