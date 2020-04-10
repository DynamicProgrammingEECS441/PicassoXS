//
//  SaveViewController.swift
//  PicassoXFinal
//
//  Created by Senhao Wang on 4/9/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import Foundation
import UIKit

class SaveViewController: UIViewController{
    
    @IBOutlet weak var Result: UIImageView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        Result.image = GlobalVariables.curResult
        GlobalVariables.FilteredHistory.insert(MadeImages(GlobalVariables.curInput!, GlobalVariables.curFilter!, GlobalVariables.curResult!), at: 0)
    }
    @IBOutlet weak var Save: UIButton!
    @IBAction func SaveToAlbum(_ sender: Any) {
        UIImageWriteToSavedPhotosAlbum(self.Result.image!, nil, nil, nil);
        Save.tintColor = GlobalVariables.Grey1
    }
    
    @IBAction func Continue(_ sender: Any) {
        self.performSegue(withIdentifier: "Home", sender: nil)
    }
}
