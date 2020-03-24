
//  DetailFilterViewController.swift
//  PicassoXs
//
//  Created by Senhao Wang on 3/23/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import Foundation
import UIKit


class DetailViewController : UIViewController{
    
    @IBAction func Apply(_ sender: Any) {
        GlobalVariables.curFilter = Image.image
        performSegue(withIdentifier: "upload", sender: self)
    }
    
    @IBOutlet weak var Name: UILabel!
    
    @IBOutlet weak var Image: UIImageView!
    
    @IBOutlet weak var Text: UITextView!
    
    override func viewDidLoad() {
        super.viewDidLoad()
        
        Name.text = GlobalVariables.Filters[DetailIndex].name
        Image.image = GlobalVariables.Filters[DetailIndex].FilterImage
        Text.text = GlobalVariables.Filters[DetailIndex].intro
        
    }
}
