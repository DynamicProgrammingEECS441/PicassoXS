//
//  HomeViewCell.swift
//  PicassoXs
//
//  Created by Senhao Wang on 3/23/20.
//  Copyright © 2020 Senhao Wang. All rights reserved.
//

import Foundation
import UIKit


class HomeViewCell: TableViewCell{
    
    @IBOutlet weak var ResultIm: UIImageView!
    @IBOutlet weak var TimeStamp: UILabel!
    
    func setHomeView(Image: MadeImages){
        ResultIm.image = Image.res
        TimeStamp.text = Image.DateTime
    }
    
    
}
