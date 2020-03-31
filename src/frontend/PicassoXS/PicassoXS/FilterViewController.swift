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

class FilterViewController : UIViewController, UITableViewDelegate, UITableViewDataSource{
    
    
    
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
        print("Failed herer!!!!!!!!!!!!!!!")
        performSegue(withIdentifier: "Detail", sender: self)
        print("Failed thererererer!!!!!!!!!!!!!!!")
        print(DetailIndex)
    }
    
    
}
